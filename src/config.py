import os
from dotenv import load_dotenv


load_dotenv()


EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


CATEGORIES = [
    "HR",
    "DESIGNER",
    "INFORMATION-TECHNOLOGY",
    "TEACHER",
    "ADVOCATE",
    "BUSINESS-DEVELOPMENT",
    "HEALTHCARE",
    "FITNESS",
    "AGRICULTURE",
    "BPO",
    "SALES",
    "CONSULTANT",
    "DIGITAL-MEDIA",
    "AUTOMOBILE",
    "CHEF",
    "FINANCE",
    "APPAREL",
    "ENGINEERING",
    "ACCOUNTANT",
    "CONSTRUCTION",
    "PUBLIC-RELATIONS",
    "BANKING",
    "ARTS",
    "AVIATION",
]


def get_env(key: str, default: str = None, required: bool = False) -> str:
    value = os.getenv(key, default)
    if required and not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return value


OPENAI_API_KEY = get_env("OPENAI_API_KEY", required=True)
PINECONE_API_KEY = get_env("PINECONE_API_KEY", required=True)
PINECONE_INDEX = get_env("PINECONE_INDEX", default="resumes-index")
PINECONE_CLOUD = get_env("PINECONE_CLOUD", default="aws")
PINECONE_REGION = get_env("PINECONE_REGION", default="us-east-1")


