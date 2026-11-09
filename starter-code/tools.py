import json
import os
from typing import List, Dict, Any
from datetime import datetime

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")

# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# TODO: Hoàn thiện hàm này — đọc file product_catalog.json, lọc theo category và max_price.
# ---------------------------------------------------------------------------


def search_product_catalog(category: str, max_price: int = 999999999999) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.
    
    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.
    
    Returns:
        Danh sách sản phẩm phù hợp điều kiện.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")

    # Kiểm tra file có tồn tại không
    if not os.path.exists(catalog_file):
        return []

    # Đọc dữ liệu từ file JSON
    with open(catalog_file, "r", encoding="utf-8") as f:
        products = json.load(f)

    # Lọc sản phẩm theo danh mục và giá tối đa
    results = []
    for p in products:
        if p["category"] == category and p["price_vnd"] <= max_price:
            results.append(p)

    return results



# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# TODO: Hoàn thiện hàm này — tạo ticket mới và lưu vào support_tickets.json.
# ---------------------------------------------------------------------------


def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.
    """
    from datetime import datetime, timezone, timedelta

    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")

    # Kiểm tra mức độ ưu tiên
    if priority not in ["low", "medium", "high"]:
        raise ValueError("Priority không hợp lệ")

    # Đảm bảo thư mục tồn tại
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # Đọc danh sách ticket cũ
    if os.path.exists(tickets_file):
        with open(tickets_file, "r", encoding="utf-8") as f:
            tickets = json.load(f)
    else:
        tickets = []

    # Lấy thời gian Việt Nam (UTC+7)
    vn_tz = timezone(timedelta(hours=7))
    now = datetime.now(vn_tz)
    today = now.strftime("%Y%m%d")

    # Tìm số thứ tự lớn nhất trong toàn bộ ticket
    seq = max(
        (
            int(t["ticket_id"].split("-")[-1])
            for t in tickets
            if t.get("ticket_id", "").split("-")[-1].isdigit()
        ),
        default=0
    ) + 1

    # Tạo ID mới
    ticket_id = f"TK-{today}-{seq:03d}"

    # Tạo ticket đúng cấu trúc JSON
    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority,
        "status": "open",
        "created_at": now.isoformat(),
        "category": "general"
    }

    # Thêm ticket mới
    tickets.append(new_ticket)

    # Lưu vào file JSON
    with open(tickets_file, "w", encoding="utf-8") as f:
        json.dump(tickets, f, ensure_ascii=False, indent=2)

    return new_ticket



    # ---------------------------------------------------------------------------
    # TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
    # TODO: Định nghĩa JSON Schema cho từng tool (name, description, parameters).
    # ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    # Tool #1: Tìm kiếm sản phẩm
    {
        "type": "function",
        "function": {
            "name": "search_product_catalog",
            "description": "Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và mức giá tối đa.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["xe_dien", "du_lich"],
                        "description": "Danh mục sản phẩm: xe điện VinFast hoặc du lịch Vinpearl."
                    },
                    "max_price": {
                        "type": "integer",
                        "description": "Giá tối đa tính bằng VNĐ. Mặc định 999999999999.",
                        "default": 999999999999
                    }
                },
                "required": ["category"],
                "additionalProperties": False
            }
        }
    },

    # Tool #2: Tạo ticket hỗ trợ
    {
        "type": "function",
        "function": {
            "name": "submit_support_ticket",
            "description": "Tạo yêu cầu hỗ trợ khách hàng và lưu vào hệ thống ticket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {
                        "type": "string",
                        "description": "Họ và tên khách hàng cần hỗ trợ."
                    },
                    "issue_description": {
                        "type": "string",
                        "description": "Mô tả chi tiết vấn đề khách hàng gặp phải."
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Mức độ ưu tiên của ticket. Mặc định medium.",
                        "default": "medium"
                    }
                },
                "required": ["customer_name", "issue_description"],
                "additionalProperties": False
            }
        }
    }
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}

