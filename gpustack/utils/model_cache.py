def validate_model_id(model_id: str) -> tuple[str, str]:
    parts = model_id.split("/")
    if len(parts) != 2 or any(not safe_path_part(part) for part in parts):
        raise ValueError("invalid_model_id")
    return parts[0], parts[1]


def model_object_prefix(root_prefix: str, model_id: str) -> str:
    organization, model_name = validate_model_id(model_id)
    root = root_prefix.strip("/")
    path = f"{organization}/{model_name}/"
    return f"{root}/{path}" if root else path


def safe_path_part(value: str) -> bool:
    return (
        bool(value)
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
        and not any(ord(char) < 32 or ord(char) == 127 for char in value)
    )
