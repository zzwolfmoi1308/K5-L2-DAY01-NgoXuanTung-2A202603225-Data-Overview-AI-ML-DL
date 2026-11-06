# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** T4 GPU

**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

2. Record này nói về cả bức ảnh. Vì model phân loại ảnh chỉ nhận vào toàn bộ ảnh và trả ra một nhãn duy nhất cho ảnh đó, không phân tích từng vùng hay từng vật thể riêng lẻ. Trường `rank` và `score` cũng xếp hạng toàn ảnh theo xác suất chung, không gắn với tọa độ hay đối tượng cụ thể nào.

3. Danh sách tên lớp đi kèm taxonomy và checkpoint. Trường `taxonomy_name = "ImageNet-1K"` cho thấy 1 000 lớp được định nghĩa sẵn trong bộ dữ liệu ImageNet-1K mà checkpoint `yolo11n-cls.pt` đã được huấn luyện trên đó, model không tự nghĩ ra tên lớp.

4. Chỉ ghi `class_name` là chưa đủ vì tên lớp có thể bị trùng hoặc có nhiều taxonomy khác nhau (ví dụ "cab" trong ImageNet-1K khác với "cab" trong một taxonomy khác). Ghi thêm `class_id` định danh duy nhất trong taxonomy đó, và `taxonomy_name` xác định rõ taxonomy nào đang dùng, tránh nhầm lẫn khi nhiều dự án dùng chung file JSON hoặc khi hợp nhất dữ liệu về sau.

5. Guideline cần nêu rõ quy tắc chọn chủ thể chính như: chọn vật thể chiếm diện tích lớn nhất / nằm trung tâm là mục tiêu chụp theo ngữ cảnh dự án. Ngoài ra cần ghi rõ cách xử lý trường hợp tranh chấp (hai vật ngang nhau) và khi nào thì escalation lên Lab Coach thay vì tự quyết.

6. `score` là độ tin của model phản ánh xác suất model đoán đúng theo phân phối xác suất nội bộ không phải sự đánh giá của con người dựa trên quy tắc gán nhãn. Ground truth phải do annotator gán theo guideline đã thống nhất. Nếu dùng `score` thay thế, nhãn sẽ bị phụ thuộc vào model hiện tại và không còn là chuẩn độc lập để đánh giá/huấn luyện model mới.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

2. Box này đóng khung một người trong ảnh bếp. Góc trên-trái nằm ở tọa độ (385.33, 69.24) Góc dưới-phải ở (498.92, 348.92) Box rộng 113.58 px và cao 279.68 px

3. So sánh hai ngưỡng (theo output notebook):
   - Ngưỡng 0.20: 17 prediction → `['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl', 'spoon', 'potted plant', 'spoon', 'dining table', 'spoon', 'bottle']`
   - Ngưỡng 0.60: 6 prediction → `['person', 'bowl', 'bowl', 'oven', 'oven', 'person']`

4. Khi hạ ngưỡng (ví dụ từ 0.35 xuống 0.20), số box tăng lên vì giữ thêm các prediction có độ tin thấp. Reviewer phải: kiểm tra thêm nhiều box hơn để loại false positive; phán xét những vật thể mờ, bị che, hoặc nhỏ mà model còn phân vân; tốn thêm thời gian QC và tỉ lệ reject có thể cao hơn, làm chậm vòng quay annotation.

5. Quy tắc box: Vẽ hộp bao sát toàn bộ phần nhìn thấy được của vật thể (bao gồm cả vùng bị che một phần), không để lề trắng thừa quá 2 px mỗi phía và không cắt bớt bất kỳ pixel nào thuộc vật thể đó. Nếu vật thể bị cắt mép ảnh, kéo cạnh box đến đúng mép ảnh.

6. Trên ảnh `kitchen`, người đứng bên trái (record `person`, score=0.610931, bbox x từ 0.12 đến 61.55) bị cắt mép trái của ảnh, chỉ thấy một phần thân người, phần còn lại nằm ngoài khung. Trường hợp này nên viết thành quy tắc trong guideline như "Nếu vật thể bị cắt bởi mép ảnh, vẫn gán nhãn và kéo cạnh box đến mép ảnh, không bỏ qua vật thể đó", vì đây là tình huống phổ biến và lặp lại trong nhiều ảnh khác. Quy tắc rõ giúp annotator xử lý nhất quán mà không cần escalation mỗi lần.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; `polygon_point_count`=348 ; 3-5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]

2. Bounding box chỉ cho biết hình chữ nhật bao ngoài còn polygon cung cấp đường viền chính xác theo hình dạng thực của vật thể. Với 348 điểm, polygon bám sát từng đường cong của người (vai, cánh tay, đầu…), giúp phân tách pixel thuộc vật thể khỏi nền hoặc các vật thể khác chạm nhau.

3. `instance_id` dùng để định danh duy nhất từng đối tượng riêng lẻ trong một ảnh, phân biệt các vật cùng lớp với nhau. Nó không phải class ID (class ID xác định loại vật thể, như `person`, `bowl`) cũng không phải tracking ID (tracking ID theo dõi cùng một vật qua nhiều frame video) `instance_id` chỉ có ý nghĩa trong phạm vi một ảnh tĩnh để gán mask riêng cho từng cá thể.

4. Quy tắc vẽ mask: Vẽ polygon bám sát mép ngoài phần nhìn thấy được của vật thể, không lấn sang vật thể khác và không bỏ sót phần lộ ra dù bị che một phần. Với các vùng bị che hoàn toàn thì không vẽ, chỉ vẽ phần pixel thực sự quan sát được trong ảnh.

5. Trên ảnh `kitchen`, cái bát (bowl) ở góc dưới-trái (instance `kitchen-003`, bbox x từ 0.32 đến 87.69) bị che khuất một phần bởi mép trái ảnh và có thể bị vật khác che phía trên. Đây là tình huống phổ biến nên viết thành quy tắc trong guideline ("vẽ polygon bao phần nhìn thấy, kéo đến mép ảnh nếu bị cắt, không cố đoán phần bị khuất") để annotator xử lý nhất quán trên mọi ảnh, không cần escalation mỗi lần gặp.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn duy nhất cho toàn ảnh gồm class_id, class_name và taxonomy_name (ví dụ class_id 468, class_name cab, taxonomy ImageNet-1K cho ảnh traffic) | Ảnh traffic có nhiều loại phương tiện như xe buýt, xe hơi và người nên khó chọn một lớp đại diện; score chỉ đạt 0.51 cho thấy model cũng không chắc chắn về nhãn | Chọn một lớp theo guideline, ví dụ vật chiếm diện tích lớn nhất hoặc là mục tiêu chính của ảnh, rồi ghi đầy đủ class_id, class_name, taxonomy_name | Kiểm tra nhãn có đúng taxonomy và đúng quy tắc chọn lớp không, phát hiện các trường hợp gán không nhất quán giữa các annotator |
| Phát hiện vật thể | Danh sách các box, mỗi box gồm class_name và bốn tọa độ pixel bbox_xyxy cho từng vật thể (ví dụ class person với tọa độ [385.33, 69.24, 498.92, 348.92] trong ảnh kitchen) | Person phía trái ảnh kitchen bị cắt mép nên ranh giới box không rõ ràng; ảnh kitchen có hai record oven khác nhau, cần xác định là hai vật riêng biệt hay bị detect trùng | Vẽ box bao sát từng vật thể, gán đúng class theo taxonomy, kéo cạnh box đến mép ảnh nếu vật bị cắt | Kiểm tra box có đủ chặt không (không lề thừa, không thiếu pixel của vật thể), không có box trùng lặp, class phải đúng theo taxonomy COCO-80 |
| Instance segmentation | Danh sách mask, mỗi instance gồm instance_id, class_name và chuỗi tọa độ polygon_xy (ví dụ kitchen-001, class person, 348 điểm polygon) | Bowl ở góc dưới bên trái ảnh kitchen bị cắt mép và bị che một phần nên khó xác định nên vẽ polygon đến đâu; hai bowl chạm nhau khó phân biệt đường biên giữa hai vật | Vẽ polygon bám sát phần nhìn thấy được của từng instance, gán instance_id riêng biệt cho từng cá thể dù cùng lớp | Kiểm tra polygon không lấn sang vật thể khác, không bỏ sót phần lộ ra, và instance_id không bị trùng trong cùng một ảnh |

## 5. An toàn dữ liệu

1. Chỉ sử dụng dữ liệu ảnh được cung cấp trong phạm vi bài lab, không tải lên hoặc chia sẻ ra ngoài, không ghi bất kỳ thông tin cá nhân nào vào báo cáo hoặc file output.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi (ví dụ ảnh chứa thông tin cá nhân, nội dung nhạy cảm hoặc không thuộc bộ dữ liệu bài lab), tôi sẽ dừng lại ngay, không xử lý tiếp và báo cáo cho Lab Coach hoặc Giảng viên hướng dẫn để được hướng dẫn xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
