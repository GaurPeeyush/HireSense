from enum import Enum
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from .config import CATEGORIES


class CategoryEnum(str, Enum):
    HR = "HR"
    DESIGNER = "DESIGNER"
    INFORMATION_TECHNOLOGY = "INFORMATION-TECHNOLOGY"
    TEACHER = "TEACHER"
    ADVOCATE = "ADVOCATE"
    BUSINESS_DEVELOPMENT = "BUSINESS-DEVELOPMENT"
    HEALTHCARE = "HEALTHCARE"
    FITNESS = "FITNESS"
    AGRICULTURE = "AGRICULTURE"
    BPO = "BPO"
    SALES = "SALES"
    CONSULTANT = "CONSULTANT"
    DIGITAL_MEDIA = "DIGITAL-MEDIA"
    AUTOMOBILE = "AUTOMOBILE"
    CHEF = "CHEF"
    FINANCE = "FINANCE"
    APPAREL = "APPAREL"
    ENGINEERING = "ENGINEERING"
    ACCOUNTANT = "ACCOUNTANT"
    CONSTRUCTION = "CONSTRUCTION"
    PUBLIC_RELATIONS = "PUBLIC-RELATIONS"
    BANKING = "BANKING"
    ARTS = "ARTS"
    AVIATION = "AVIATION"


class SelectCategory(BaseModel):
    category: CategoryEnum


_client = OpenAI()


def classify_category(user_query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a classifier. Respond ONLY with a JSON object of the form {\"category\": \"...\"}. "
                "Return exactly one category that best matches the user's query. "
                "Choose only from the allowed enum categories. No extra text."
            ),
        },
        {
            "role": "user",
            "content": (
                "Allowed categories: " + ", ".join(CATEGORIES) + "\n\n" + user_query
            ),
        },
    ]
    response = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content
    # pydantic parsing for safety
    obj = SelectCategory.model_validate_json(content)
    return obj.category.value