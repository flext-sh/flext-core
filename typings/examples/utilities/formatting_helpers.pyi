from flext_core import TEntityId as TEntityId

SECONDS_PER_MINUTE: int
SECONDS_PER_HOUR: int
GRADE_A_THRESHOLD: int
GRADE_B_THRESHOLD: int
BYTES_PER_KB: int
YOUNG_ADULT_AGE_THRESHOLD: int
ADULT_AGE_THRESHOLD: int
MIDDLE_AGED_THRESHOLD: int
MAX_DISCOUNT_PERCENTAGE: int

def generate_prefixed_id(prefix: str, length: int) -> TEntityId: ...
def generate_hash_id(data: str) -> TEntityId: ...
def generate_short_id(length: int = 8) -> TEntityId: ...
def get_age_category(age_value: int) -> str: ...
