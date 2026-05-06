# ask-hasabot-meeting

Repo này là một backend API nhỏ cho tính năng **Ask Hasabot** trong ngữ cảnh cuộc họp.

Sau khi cuộc họp đã được tổng hợp, người dùng có thể chọn một đoạn trong phần gợi ý giao việc, đặt câu hỏi về đoạn đó, rồi backend sẽ gửi câu hỏi cùng ngữ cảnh liên quan cho LLM để tạo câu trả lời.

## Repo Này Làm Gì?

Repo này không tạo meeting summary từ đầu.

Nó giả định rằng frontend hoặc hệ thống khác đã có sẵn:

- nội dung cuộc họp đã được tóm tắt
- gợi ý giao việc
- đoạn người dùng chọn
- câu hỏi người dùng hỏi về đoạn đó

NHIỆM VỤ CHÍNH của repo là hỗ trợ người dùng hỏi Hasabot khi có một đoạn trong gợi ý giao việc chưa rõ.

Cụ thể, khi người dùng chọn một đoạn nội dung và đặt câu hỏi, hệ thống sẽ dùng thông tin cuộc họp, nội dung report và gợi ý giao việc hiện tại để tạo câu trả lời phù hợp. Câu trả lời sẽ được gửi dần về giao diện để người dùng có thể đọc trong lúc Hasabot đang phản hồi.

Nói ngắn gọn:

Đây là API backend giúp người dùng làm rõ một đoạn được chọn trong gợi ý giao việc sau cuộc họp.

## Luồng Xử Lý Tổng Quan

Luồng chính của repo là:

```text
Frontend hoặc curl
  -> main.py nhận request từ client
  -> meeting_context.py chuẩn bị ngữ cảnh và prompt cho LLM
  -> suggestion.py gọi LLM agent để xử lý câu hỏi
  -> LLM tạo câu trả lời dựa trên ngữ cảnh đã cung cấp
  -> main.py stream câu trả lời về lại client
```

## Vai Trò Từng File

### `main.py`

Đây là lớp API.

File này định nghĩa endpoint:

```text
POST /api/v2/meeting-summary/assignment-suggestion/ask
```

Nó chịu trách nhiệm:

- nhận request từ frontend
- validate các field bắt buộc như `session_id`, `event_id`, `suggestion`, `selected_text`, `question_text`
- gọi service trong `meeting_context.py`
- stream kết quả trả về cho client

Hai field quan trọng nhất trong request là:

```text
selected_text = đoạn người dùng chọn
question_text = câu hỏi người dùng hỏi về đoạn đó
```

### `meeting_context.py`

Đây là file trung tâm nếu muốn hiểu phần prompt.

File này chịu trách nhiệm:

- lấy dữ liệu từ `main.py`
- build system prompt cho LLM
- build context prompt từ meeting context, selected text và question
- gọi agent trong `suggestion.py`

Hai function quan trọng:

```python
_get_assignment_suggestion_ask_prompt(...)
```

Function này tạo prompt điều khiển hành vi của model.

```python
_build_assignment_suggestion_context_input(...)
```

Function này tạo phần context thực tế mà model sẽ đọc

### `suggestion.py`

Đây là lớp kết nối với LLM.

File này chịu trách nhiệm:

- cấu hình model
- tạo LLM agent
- gửi prompt/context vào model
- nhận output dạng stream
- trả từng phần output về lại `meeting_context.py`

Nói đơn giản:

```text
meeting_context.py quyết định gửi gì cho LLM
suggestion.py thực hiện việc gọi LLM
```

### `utils.py`

Đây là file helper.

Nó chứa các function để:

- clean text
- normalize meeting metadata
- normalize topic/detail/event
- hỗ trợ format dữ liệu trước khi đưa vào context

Đây không phải file nên đọc đầu tiên. Nên đọc sau khi đã hiểu `main.py`, `meeting_context.py`, và `suggestion.py`.

### `run_server.py`

File này được thêm để chạy local server cho dễ.

Lý do cần file này là vì `main.py` chỉ khai báo:

```python
router = APIRouter()
```

Nó không tạo sẵn một object:

```python
app = FastAPI()
```

Vì vậy `run_server.py` tạo một FastAPI app tạm, include router từ `main.py`, rồi chạy server bằng uvicorn.

## Cách Chạy Local Server

Kích hoạt môi trường Python 3.12:

```bash
conda activate ask-hasabot-py312
```

Chạy server:

```bash
python run_server.py
```

Server sẽ chạy ở:

```text
http://127.0.0.1:8000
```

Trang docs tự động của FastAPI nằm ở:

```text
http://127.0.0.1:8000/docs
```

## Cách Test API Bằng Curl

Mở terminal thứ hai và gửi POST request:

```bash
curl -N -X POST "http://127.0.0.1:8000/api/v2/meeting-summary/assignment-suggestion/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session",
    "event_id": "test-event",
    "language": "vi",
    "topic": "Cập nhật tiến độ dự án robot hút bụi",
    "report_detail": "Team đang phát triển robot hút bụi tự động cho công ty. Trong tuần này cần hoàn thiện tính năng tránh vật cản và quay về dock sạc sau khi pin yếu.",
    "detail_points": [
      "Robot đôi khi bị kẹt dưới ghế thấp.",
      "Cảm biến phía trước chưa nhận diện tốt dây điện nhỏ.",
      "Khi pin dưới 15%, robot cần tự quay về dock sạc.",
      "QA cần chuẩn bị phòng test với bàn, ghế, dây điện và thảm."
    ],
    "summary": "Team cần cải thiện khả năng tránh vật cản và cơ chế tự quay về dock sạc để chuẩn bị demo robot hút bụi vào cuối tuần.",
    "task_description": "Làm rõ yêu cầu tránh vật cản và quay về dock sạc cho robot hút bụi.",
    "suggestion": "Nhóm điều hướng cần cập nhật thuật toán để robot phát hiện ghế thấp, dây điện và tự quay về dock khi pin yếu.",
    "selected_text": "robot phát hiện ghế thấp, dây điện và tự quay về dock khi pin yếu",
    "question_text": "Cách thức để giải quyết vấn đề này là gì?"
  }'
```

Ở đây:

```text
curl = giả lập frontend
POST API = endpoint Ask Hasabot
JSON body = dữ liệu mà frontend thật sẽ gửi
```

## Prompt Trong Repo Nằm Ở Đâu?

Prompt chính nằm trong `meeting_context.py`.

Function quan trọng nhất là:

```python
_get_assignment_suggestion_ask_prompt(language: str) -> str
```

Đây là system/behavior prompt, dùng để điều khiển model trả lời như thế nào.

Khi gọi API bằng curl, prompt này là prompt chính ảnh hưởng đến output của LLM.

Context runtime được build bởi:

```python
_build_assignment_suggestion_context_input(...)
```

Function này gom các field từ request thành một context có cấu trúc để model đọc.

## Prompt Gốc Và Modified Prompt Khác Nhau Gì?

Trước khi có `# The modified prompt`, prompt gốc cho model khá linh hoạt.

Model được yêu cầu:

- trả lời trực tiếp câu hỏi của người dùng
- giải thích đoạn được chọn trong ngữ cảnh toàn bộ suggestion
- có thể viết lại đoạn được chọn cho rõ hơn
- có thể đưa gợi ý thực thi, rủi ro, ưu tiên, bước tiếp theo
- dùng markdown khi cần

Output khi đó có thể dài/ngắn tùy câu hỏi.

Sau khi có `# The modified prompt`, prompt được chuyển sang dạng few-shot cứng hơn.

Với tiếng Việt, model bị ép:

- trả lời đúng 3 bullet
- mỗi bullet bắt đầu bằng `Bằng chứng:`
- mỗi bullet chỉ là một câu
- chỉ dùng context được cung cấp
- không thêm giải thích trước hoặc sau 3 bullet

Với tiếng Anh, model bị ép:

- trả lời đúng 3 bullet
- mỗi bullet bắt đầu bằng `Proof:`
- mỗi bullet chỉ là một câu
- không thêm phần mở đầu hoặc kết luận

Mục tiêu của modified prompt là test kỹ thuật **few-shot prompting** và **structured output prompting**.

Nói đơn giản:

```text
Prompt gốc = linh hoạt, tư vấn rộng hơn
Modified prompt = output cứng, đúng format, tập trung vào bằng chứng
```

## Prompt Engineering Trong Repo

Nếu muốn thử prompt engineering, điểm bắt đầu là:

```python
_get_assignment_suggestion_ask_prompt(...)
```

Nếu muốn thay đổi cách context được trình bày cho model, xem:

```python
_build_assignment_suggestion_context_input(...)
```

Nếu muốn thay đổi model config như temperature, top_p, max_output_tokens, xem:

```python
suggestion.py
_build_generation_config()
```
