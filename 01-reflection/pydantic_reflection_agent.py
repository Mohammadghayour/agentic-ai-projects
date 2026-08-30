"""
Self-Correcting JSON & Pydantic Guard Agent (Reflection Pattern)
===============================================================
الگوی بازاندیشی (Reflection) در هوش مصنوعی عاملی
این اسکریپت نشان می‌دهد چگونه می‌توان با اتصال بازخورد خارجی کتابخانه Pydantic،
خطاهای معنایی و فرمتی خروجی مدل‌های زبانی (LLM) را به صورت خودکار اصلاح کرد.
"""

import os
import sys
import json
import re
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, ValidationError
from openai import OpenAI

# تنظیم خروجی کنسول روی UTF-8 برای پشتیبانی از کاراکترهای فارسی و ایموجی‌ها در ویندوز
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# بارگذاری متغیرهای محیطی از فایل .env
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# راه‌اندازی کلاینت با پشتیبانی از Base URL سفارشی
client = OpenAI(
    api_key=API_KEY if API_KEY and API_KEY != "your_api_key_here" else "dummy_key",
    base_url=BASE_URL
)


# =====================================================================
# ۱. تعریف ساختار استاندارد داده با Pydantic (قوانین سخت‌گیرانه بیزینس)
# =====================================================================
class InvoiceData(BaseModel):
    customer_name: str = Field(..., description="نام کامل مشتری")
    email: str = Field(..., description="ایمیل معتبر مشتری")
    total_amount: float = Field(..., description="مبلغ کل فاکتور که باید حتماً مقداری مثبت باشد")
    items_count: int = Field(..., ge=1, description="تعداد اقلام خریداری شده (حداقل ۱ عدد)")
    discount_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="درصد تخفیف بین ۰ تا ۱۰۰")

    @field_validator("email")
    def validate_email_format(cls, value: str) -> str:
        """اعتبارسنجی فرمت ایمیل"""
        email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        if not re.match(email_regex, value.strip()):
            raise ValueError(f"ایمیل '{value}' فرمت معتبری ندارد (مثال صحیح: user@example.com).")
        return value.strip()

    @field_validator("total_amount")
    def validate_positive_amount(cls, value: float) -> float:
        """اطمینان از مثبت بودن مبلغ کل فاکتور"""
        if value <= 0:
            raise ValueError(f"مبلغ کل فاکتور ({value}) نمی‌تواند صفر یا عدد منفی باشد.")
        return value


# تابع کمکی برای پاک‌سازی خروجی JSON از تگ‌های مارک‌داون
def clean_json_string(raw_response: str) -> str:
    cleaned = raw_response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


# =====================================================================
# ۲. گام اول: تولید اولیه پیش‌نویس (Generator - V1)
# =====================================================================
def generate_v1(raw_text: str) -> str:
    """
    استخراج اولیه داده به صورت JSON از متن غیرساختاریافته کاربر.
    """
    prompt = f"""
    You are an AI data extractor. Extract structured information from the following unformatted text into JSON.

    Target JSON Fields:
    - customer_name (string)
    - email (string)
    - total_amount (number)
    - items_count (integer)
    - discount_percentage (number, default 0)

    Unformatted Input Text:
    \"\"\"{raw_text}\"\"\"

    Return ONLY strict JSON with no additional explanation or markdown.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return clean_json_string(response.choices[0].message.content)


# =====================================================================
# ۳. گام دوم: اعتبارسنجی در محیط واقعی با Pydantic (External Feedback)
# =====================================================================
def validate_with_pydantic(json_str: str) -> tuple[bool, str]:
    """
    بررسی صحت خروجی JSON با قوانین مدل Pydantic.
    خروجی: (آیا معتبر است، نتیجه موفق یا پیام خطا)
    """
    try:
        data = json.loads(json_str)
        validated_obj = InvoiceData(**data)
        return True, validated_obj.model_dump_json(indent=2)
    except json.JSONDecodeError as json_err:
        return False, f"JSON Syntax Error: {str(json_err)}"
    except ValidationError as val_err:
        # استخراج خطاهای خوانا و مشخص
        error_details = []
        for err in val_err.errors():
            field = " -> ".join([str(loc) for loc in err["loc"]])
            msg = err["msg"]
            error_details.append(f"- Field '{field}': {msg}")
        return False, "\n".join(error_details)


# =====================================================================
# ۴. گام سوم: بازاندیشی و اصلاح خروجی (Reflection & Refiner - V2)
# =====================================================================
def reflect_and_refine(raw_text: str, failed_json: str, feedback_errors: str) -> str:
    """
    ارسال بازخورد خطاهای Pydantic به مدل ارزیاب برای اصلاح و تولید نسخه V2.
    """
    prompt = f"""
    You are a data validation and self-correction assistant (Reflection Pattern).

    A previously extracted JSON failed strict Pydantic validation rules.

    Original Text:
    \"\"\"{raw_text}\"\"\"

    Failed Draft JSON (V1):
    {failed_json}

    Validation Errors Encountered:
    {feedback_errors}

    Instructions:
    1. Carefully analyze each validation error against the original text.
    2. Correct the values (e.g. fix email typos/formats, ensure positive values, verify ranges).
    3. Return ONLY the corrected, valid JSON.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return clean_json_string(response.choices[0].message.content)


# =====================================================================
# ۵. پایپ‌لاین کلی بازاندیشی (End-to-End Workflow)
# =====================================================================
def run_reflection_pipeline(raw_text: str):
    print("=" * 70)
    print(" شروع فرآیند عامل هوشمند خوداصلاحگر (Pydantic Reflection Agent)")
    print("=" * 70)

    print("\n [ورودی خام]:")
    print(raw_text.strip())

    # گام اول: تولید V1
    print("\n" + "-" * 70)
    print(" [گام ۱: تولید اولیه پیش‌نویس - V1 Generator]")
    v1_output = generate_v1(raw_text)
    print(v1_output)

    # گام دوم: اعتبارسنجی با محیط
    print("\n" + "-" * 70)
    print(" [گام ۲: اعتبارسنجی در محیط با Pydantic]")
    is_valid, feedback = validate_with_pydantic(v1_output)

    if is_valid:
        print(" خروجی V1 در تلاش اول کاملاً معتبر بود:")
        print(feedback)
        return

    print(" خطای اعتبارسنجی کشف شد (External Feedback):")
    print(feedback)

    # گام سوم: بازاندیشی و اصلاح V2
    print("\n" + "-" * 70)
    print(" [گام ۳: بازاندیشی و اصلاح خطاها بر مبنای فیدبک - V2 Refiner]")
    v2_output = reflect_and_refine(raw_text, v1_output, feedback)
    print(v2_output)

    # گام چهارم: اعتبارسنجی نهایی
    print("\n" + "-" * 70)
    print(" [گام ۴: تایید نهایی خروجی اصلاح‌شده V2]")
    is_valid_v2, final_result = validate_with_pydantic(v2_output)

    if is_valid_v2:
        print("✅ موفقیت کامل! داده‌ها با رعایت ۱۰۰٪ قوانین اسکیما استخراج شدند:")
        print(final_result)
    else:
        print("⚠️ نیاز به تکرار مجدد چرخه بازاندیشی:")
        print(final_result)

    print("\n" + "=" * 70)


# =====================================================================
# نمونه داده آزمایشی برای اجرای تست
# =====================================================================
if __name__ == "__main__":
    # یک متن غیررسمی با چند چالش ظریف:
    # ۱. ایمیل با فرمت اشتباه نوشته شده (at به جای @ یا فاقد دامنه)
    # ۲. مقدار کل با منفی درج شده (به دلیل مرجوعی یا کسر حساب)
    # ۳. تعداد آیتم‌ها و درصد تخفیف در متن پراکنده است
    sample_raw_invoice = """
    سلام، سفارش آقای علی رضایی ثبت شد.
    ایمیل ارتباطی ایشان ali.rezaei[at]gmail_com است.
    فاکتور شامل ۳ عدد محصول به مبلغ نهایی کل منفی ۱۵۰ هزار تومان (-150.0) به دلیل بستانکاری قبلی است
    که البته ارزش واقعی کل فاکتور برای ثبت سیستم همان مثبت ۱۵۰ دلار/تومان باید لحاظ شود.
    تخفیف اعمال شده روی این خرید ۱۰ درصد بوده است.
    """

    if not API_KEY or API_KEY == "your_api_key_here":
        print("⚠️ لطفاً ابتدا فایل .env را باز کرده و OPENAI_API_KEY خود را در آن وارد کنید.")
    else:
        run_reflection_pipeline(sample_raw_invoice)
