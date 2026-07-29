# **Báo cáo Nghiên cứu Chuyên sâu: Chiến lược Rút ngắn Khoảng cách Hiệu năng 3D Gaussian Splatting và Giải pháp SOTA**

## **Quyết định Điều hành và Định vị Bài toán**

Báo cáo này tập trung phân tích toàn diện lộ trình kỹ thuật nhằm nâng điểm số của hệ thống tái tạo không gian 3D (Novel View Synthesis) từ 75.3793 lên mục tiêu 77 trên tập dữ liệu Vòng 2\. Quá trình tái tạo 3D này sử dụng 3D Gaussian Splatting (3DGS) để tạo ra các "bản sao kỹ thuật số" (digital twin) có độ trung thực quang học cao. Dựa trên dữ liệu đánh giá, lỗi còn lại không phải là vấn đề về dung lượng hay sức mạnh tổng thể của mô hình (global capacity problem). Năm cảnh quay bằng drone đã đạt mức 77.2–77.7; do đó, gần như toàn bộ dư địa để cải thiện (headroom) nằm ở hai cảnh quay trong nhà: bonsai và chair.  
Để nâng tổng điểm của bảy cảnh trên bảng xếp hạng (leaderboard) từ 75.3793 lên 77, chúng ta cần đạt được mức tăng tổng cộng khoảng \+0.11345 điểm cho hai cảnh trong nhà này (giả sử điểm số của drone giữ nguyên), tương đương khoảng \+0.0567 điểm mỗi cảnh. Đánh giá sơ bộ khẳng định rằng việc tinh chỉnh các siêu tham số (hyperparameter polishing) sẽ hoàn toàn không đủ khả năng lấp đầy khoảng trống này. Thay vào đó, chúng ta phải đối mặt với những giới hạn cốt lõi về mặt kiến trúc và mô hình hóa vật lý.  
Hai khoảng trống chủ đạo (dominant gaps) đã được xác định bao gồm:

> 1. **Cảnh bonsai**: Dữ liệu nguồn không có tính dừng về mặt không gian (spatially non-stationary). Việc camera quay vòng quanh vật thể nhiều lần khiến cho chiến lược lựa chọn nguồn dựa trên khoảng cách trung tâm (center-distance source selection) bị "aliasing" – gán nhầm các góc nhìn cách xa nhau hàng trăm khung hình thành các góc nhìn tham chiếu lân cận. Tuy nhiên, việc thay thế cơ chế này bằng một bảng xếp hạng góc nhìn/thời gian toàn cục (global temporal/pose ranking) lại phá vỡ cấu trúc che khuất (occlusion) và làm giảm mạnh chỉ số LPIPS. Sự thay đổi cốt lõi cần thiết ở đây là **cơ chế chú ý/độ tin cậy của nguồn học được trên từng điểm ảnh (per-pixel learned source confidence/attention)**, với trình kết xuất (renderer) đóng vai trò là một lớp dự phòng (fallback) tường minh.  
> 2. **Cảnh chair**: Chất lượng hình ảnh thu thập được trở nên mờ đi đáng kể theo thời gian quay (substantially blurrier through time). Đây hoàn toàn là bài toán **tái tạo nhận thức độ nhòe (blur-aware reconstruction)**, không phải là vấn đề về số lượng điểm lân cận hay thiếu hụt dữ liệu tinh chỉnh (refiner-data). Thử nghiệm backbone tiếp theo bắt buộc phải tích hợp một quỹ đạo phơi sáng ngắn, được điều chuẩn hóa (short, regularized exposure trajectory) trong khi vẫn giữ cố định điểm giữa của camera (midpoint camera) được cung cấp.

Đáng chú ý, ba thay đổi có vẻ hấp dẫn đã được thử nghiệm và bị loại bỏ trên một tập holdout bên ngoài (clean outer holdout) hoàn toàn độc lập: gộp nhóm nguồn thời gian toàn cục (global temporal source bracketing), xếp hạng nguồn nhận thức góc nhìn toàn cục (global pose-aware source ranking), và việc thay thế trực tiếp thành phần L1 của bộ tinh chỉnh (refiner) bằng PSNR chuẩn hóa. Những yếu tố này tuyệt đối không được đưa vào bản đệ trình cuối cùng nếu chỉ dựa trên các bằng chứng proxy gián tiếp.

## **Tổng quan Kiến thức về 3D Gaussian Splatting dành cho Kỹ sư Học máy**

Đối với các kỹ sư học máy (Machine Learning Engineer) tại Việt Nam – những người đã quen thuộc với Computer Vision truyền thống như Mạng nơ-ron tích chập (CNNs), Vision Transformers (ViTs), hay xử lý ảnh 2D – việc chuyển sang không gian 3D yêu cầu một sự thay đổi trong tư duy biểu diễn. Trước đây, NeRF (Neural Radiance Fields) thống trị lĩnh vực này bằng cách sử dụng mạng Multi-Layer Perceptron (MLP) để mô hình hóa hàm bức xạ và mật độ tại một điểm tọa độ không gian 5D liên tục. Tuy nhiên, NeRF gặp điểm yếu chí mạng về tốc độ do phải nội suy tia (ray marching) dày đặc.  
3D Gaussian Splatting (3DGS) giải quyết bài toán này bằng cách biểu diễn cảnh 3D bằng một tập hợp các hạt rời rạc mang hình dáng phân phối chuẩn Gauss (3D Gaussians). Mỗi hạt Gauss sở hữu các tham số học được (learnable parameters) bao gồm:

> * **Vị trí (![][image1])**: Tọa độ 3D trung tâm (X, Y, Z).  
> * **Hiệp phương sai (![][image2])**: Quyết định hình dáng và kích thước của hạt (được tối ưu thông qua Quaternion cho phép xoay và Vector cho phép co giãn 3 trục).  
> * **Độ mờ đục (![][image3])**: Mức độ che khuất ánh sáng.  
> * **Màu sắc (Color)**: Biểu diễn qua các hệ số Spherical Harmonics (SH) để màu sắc có thể thay đổi tùy thuộc vào góc nhìn của camera (view-dependent).

Quá trình "Differentiable Rasterization" (Kết xuất phân giải vi phân) sẽ chiếu (splat) các hạt 3D này xuống mặt phẳng 2D, sắp xếp chúng theo chiều sâu (depth sorting) và trộn alpha (alpha-blending) cực kỳ tối ưu. Mặc dù 3DGS đạt được tốc độ kết xuất thời gian thực, nhưng vì nó là một hệ thống rời rạc, không bị ràng buộc bởi một cấu trúc bề mặt liên tục (như Mesh), nó rất dễ sinh ra các điểm nhiễu (floaters), hoặc phản ứng tiêu cực khi dữ liệu đầu vào chứa các khung hình bị nhòe (motion blur) hoặc ánh sáng thay đổi (aliasing). Các giải pháp phân tích dưới đây sẽ tập trung khắc phục nhược điểm cấu trúc này.

## **Độ tin cậy của Kho lưu trữ và Tính tái lập**

Việc duy trì tính nhất quán trong môi trường thực nghiệm là tối quan trọng để các biến đổi kiến trúc không bị nhiễu bởi lỗi cơ sở dữ liệu.

> * Nhánh đang hoạt động (Active branch): hai\_dev.  
> * Nhánh hai\_dev cục bộ và origin/hai\_dev đã được fetch hoàn toàn giống nhau tại mã commit a5bf966590fdb1fc9ece854d0702bfcbc5c6b1eb vào ngày 2026-07-29.  
> * Nhánh origin/master đã fetch là 8e16cb6 và đã được gộp vào hai\_dev.  
> * Nhánh master cục bộ hiện đang cũ (f72a9e7, trễ 21 commit so với origin/master), do đó tuyệt đối không được sử dụng nhánh này cho các thử nghiệm tiếp theo.  
> * Việc kiểm tra xác thực GitHub trực tiếp (live authentication check) không khả dụng cho remote riêng tư này; tuyên bố trên dựa trên các tham chiếu theo dõi từ xa đã được fetch.  
> * Kho lưu trữ hiện tại không chứa các checkpoint 30k điểm sản xuất (production 30k checkpoints) hoặc mây điểm dày đặc (dense point clouds). Do đó, một checkpoint bonsai khởi tạo thưa thớt 10k điểm không rò rỉ (leak-free 10k sparse-init) đã được huấn luyện riêng để phục vụ các thử nghiệm đối chiếu bên dưới. Các điểm số thí điểm tuyệt đối (absolute pilot scores) không phải là điểm số sản xuất; nhưng các chênh lệch đối chiếu (matched deltas) là hoàn toàn hợp lệ để đánh giá kiến trúc.

Cấu trúc thư mục giai đoạn 2 (phase-2 layout) được hỗ trợ trực tiếp và thông qua liên kết tương thích được tài liệu hóa: data/raw/phase2/round2 \-\> ..

## **Phân tích Bằng chứng và Khoảng trống Hiện tại**

### **Sự tập trung của Điểm số (Score Concentration)**

Chỉ số đánh giá chính thức (official metric) của mô hình được tính như sau:  
Score \= 0.4 \* (1 \- LPIPS\_VGG) \+ 0.3 \* SSIM \+ 0.3 \* clip(PSNR / 50, 0, 1\)  
Điều quan trọng đối với kỹ sư Computer Vision là nhận ra trọng số cao nhất (0.4) được dành cho LPIPS (Learned Perceptual Image Patch Similarity). Khác với PSNR chỉ đo lường sai số bình phương trung bình (MSE) ở cấp độ pixel, LPIPS sử dụng mạng VGG để trích xuất đặc trưng hình ảnh và so sánh cấu trúc nhận thức, cấu trúc không gian và kết cấu (texture). Mức tăng điểm của luồng drone v7a hầu như chỉ đến từ sự cải thiện của LPIPS. Điều đó biến việc bảo toàn kết cấu (texture preservation) và duy trì độ sắc nét của thông tin tần số cao (high-frequency evidence) nhất quán qua các góc nhìn trở nên có giá trị hơn rất nhiều so với những cải thiện nhỏ chỉ tác động tới PSNR.  
Dưới đây là hiệu năng hiện tại trên các tập holdout trong nhà không bị rò rỉ dữ liệu (leak-free indoor holdouts):

| Giai đoạn (Stage) | bonsai | chair |
| :---- | :---- | :---- |
| Chỉ sử dụng Backbone (Backbone only) | 0.6759 | 0.6506 |
| Full stack trước đó (Earlier full stack) | 0.6913 | 0.6650 |
| Nhánh tốt nhất được ghi nhận (Best recorded branch) | 0.7066 (SSS gate) | 0.6724 |

Điểm đáng chú ý từ bảng trên là việc áp dụng cổng Subsurface Scattering (SSS gate \- Tán xạ dưới bề mặt) giúp tăng mạnh điểm số cho cảnh bonsai (vật thể thực vật có độ thấu quang) nhưng lại làm giảm điểm của cảnh chair (bề mặt bàn bóng loáng, không thấu quang). Điều này chứng minh một cách rõ ràng: một backbone duy nhất áp dụng chung cho cả hai cảnh trong nhà là không phù hợp.

### **Bonsai: Thất bại của Cơ chế Lựa chọn Nguồn Hiện tại**

Trong một hệ thống học đa góc nhìn (Multi-View Stereo \- MVS) hoặc Image-Based Rendering (IBR), thuật toán cần chọn ra ![][image4] hình ảnh lân cận (nguồn) để nội suy hoặc tham chiếu kết cấu cho góc nhìn hiện tại. Chẩn đoán không cần huấn luyện (zero-training diagnostic) dưới đây thực hiện bằng cách loại bỏ từng ảnh huấn luyện được cung cấp và dự đoán độ sáng/độ sắc nét của nó từ K=3 nguồn được chọn. Thuật toán này không bao giờ đọc dữ liệu Ground Truth (GT) của tập test.

| Chính sách K=3 (K=3 policy) | MAE độ sáng (brightness MAE) | MAE độ sắc nét (sharpness MAE) | rho sắc nét (sharpness rho) | time-gap p90 train | time-gap p90 pose ẩn | test poses được bracket |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| spatial (hiện tại) | 2.491 | 0.203 | 0.761 | 785 | 1001 | 18/28 |
| pose-aware | 1.689 | 0.192 | 0.802 | 481 | 1027 | 20/28 |
| temporal bracket | 1.606 | 0.193 | 0.800 | 50 | 75 | 28/28 |

Bảng trên đã chứng minh sự tồn tại của hiện tượng aliasing thời gian (temporal alias): khi chọn theo không gian (spatial), thuật toán có xu hướng kéo các frame cách xa nhau (time-gap p90 lên tới 785/1001 frames), dẫn tới sự chênh lệch lớn về độ sáng (MAE \= 2.491). Việc chuyển sang chọn theo khung thời gian (temporal bracket) giảm time-gap xuống 50-75 và cải thiện MAE độ sáng đáng kể (1.606).  
Tuy nhiên, bài thử nghiệm A/B trên toàn bộ trình kết xuất lại bác bỏ giải pháp ngây thơ này:

| Chính sách (Policy) | Điểm (Score) | PSNR | SSIM | LPIPS | Delta so với spatial |
| :---- | :---- | :---- | :---- | :---- | :---- |
| spatial | **0.658455** | 25.2599 | 0.810170 | **0.340389** | — |
| pose-aware | 0.655768 | 25.2095 | 0.808225 | 0.344893 | **\-0.002687** |
| temporal bracket | 0.655311 | 25.2446 | 0.807965 | 0.346366 | **\-0.003144** |

Kết quả cho thấy, nhánh "temporal" chỉ thắng ở 6 frame nhưng thua ở 14 frame; nhánh "pose" thắng 11 và thua 14\. Mặc dù một vài frame cải thiện mạnh (frame\_001440, frame\_001500), các frame như frame\_001040, frame\_001860, frame\_001900, frame\_002580, và frame\_002610 lại gặp phải sự suy giảm LPIPS nghiêm trọng (LPIPS cao là tệ). Đây chính là dấu hiệu điển hình của một bộ chọn lọc (selector) hoạt động đúng ở mức độ toàn bộ bức ảnh (image level) nhưng lại hoàn toàn sai lầm ở từng bề mặt cụ thể (individual surfaces) do sự thay đổi của che khuất (occlusion).  
Hơn thế nữa, DIBR (Depth-Image-Based Rendering) hiện hành báo cáo tỷ lệ "fallback" (chuyển sang dùng dự phòng của 3DGS thay vì ảnh nguồn) lên tới 38–56% trên các góc nhìn sớm khó khăn của cảnh bonsai, trong khi cơ chế bảo vệ quang học (photometric guard) của nó chỉ từ chối khoảng 1% các mẫu có độ sâu nhất quán. Điều này khẳng định cơ chế giới hạn hệ thống chính là sự hiển thị của nguồn (source visibility) và độ bao phủ độ sâu (depth coverage), chứ không phải do ngưỡng bảo vệ quá khắt khe.

### **Giải pháp Kiến trúc cho Bonsai: Per-pixel Learned Source Attention (P1)**

Vì các phép gộp toàn cục (global logic) đã thất bại, chúng ta cần một cơ chế nội suy học sâu tương tự như IBRNet1. Trong IBRNet, mạng không cố định một chiến lược lấy mẫu toàn cục. Thay vào đó, nó trích xuất các đặc trưng 2D từ nguồn (source views) dọc theo tia (ray) và dùng một Transformer hoặc MLP để tự động đánh giá mức độ đóng góp (attention) của mỗi nguồn cho điểm ảnh đó1. Cấu trúc này không yêu cầu hình học ủy quyền (proxy geometry) hoàn hảo1.  
Thay vì loại bỏ K=3 trên phạm vi toàn cầu, lộ trình mới sẽ xây dựng một tập hợp ứng viên (candidate pool) gồm K=6–8 góc nhìn tiềm năng (chọn theo cả spatial, pose, temporal). Sau đó, một mạng nơ-ron nhẹ (lightweight network) sẽ được huấn luyện để phân bổ trọng số Softmax trên từng pixel một (per-pixel), kèm theo một trọng số dự phòng (fallback weight) dành riêng cho renderer của 3DGS. Các kênh đặc trưng (channels) đầu vào cho mỗi ứng viên sẽ bao gồm:

> * Giá trị RGB đã được chiếu warping (warped RGB).  
> * Sự bất đồng về độ sâu tương đối (relative depth disagreement) và cờ trạng thái hợp lệ/bị che khuất (valid/occluded flags).  
> * Độ tin cậy của phép chiếu ngược (reprojection confidence).  
> * Khoảng cách tâm máy ảnh chuẩn hóa (normalized camera-center distance).  
> * Độ chênh lệch góc nhìn (viewing-angle difference).  
> * Khoảng cách thời gian khung hình tuyệt đối/có dấu (signed/absolute frame-time gap).  
> * Độ sắc nét của nguồn và độ chênh lệch phơi sáng (source sharpness and exposure difference).

Để giải quyết tình trạng thiếu VRAM, cấu hình ban đầu có thể giữ K=5. Tensor bằng chứng hiện tại sẽ có kích thước 7 \+ 4K \+ 1 \= 28 kênh, hoàn toàn an toàn trên bộ nhớ 16 GB khi sử dụng 256 crops, patch base 32–48 và batch size là 2\. Việc áp dụng chuẩn hóa Entropy và Total-Variation (TV) lên các trọng số Softmax, kết hợp cùng kỹ thuật source dropout (loại bỏ nguồn ngẫu nhiên) sẽ ngăn chặn mạng lưới học vẹt (memorizing) ID của ảnh. Cấu trúc học cần tuân theo nguyên tắc "leave-one-view-out" và tuyệt đối không để lộ các pixel của tập test ẩn.

### **Chair: Độ mờ phụ thuộc vào Thời điểm Ghi hình (Capture-dependent Blur)**

Đối với cảnh chair, các chẩn đoán hiện hành chỉ ra sự suy giảm mạnh về độ sắc nét tại thời điểm chụp và sự xuống cấp tồi tệ của chỉ số LPIPS ở các khung hình cuối. Việc cố gắng thay đổi số lượng cặp tinh chỉnh (refiner pair count) hoặc lựa chọn các cặp ảnh có phân tầng sắc nét đều chỉ giữ tổng điểm quanh mức 0.664–0.666. Điều này đóng sập cánh cửa cho ý tưởng "nhiều/chất lượng dữ liệu refiner tốt hơn sẽ giải quyết vấn đề".  
Trong Computer Vision truyền thống, đối mặt với video bị mờ, kỹ sư thường nghĩ đến nội suy khung hình (Frame Interpolation) hoặc dùng mạng Deblur. Tuy nhiên, phép nội suy điểm giữa sử dụng RIFE v4.25 (một SOTA về nội suy 2D) đã thất bại thảm hại:

| Cảnh (Scene) | RIFE khắt khe (strict RIFE) | Renderer kiểm soát (renderer control) |
| :---- | :---- | :---- |
| chair | 0.4243 | 0.6521 |
| bonsai | 0.5313 | 0.6787 |

Nguyên nhân thất bại là vì RIFE chỉ nội suy chuyển động 2D giữa các pixel mà không hề hiểu về hình học 3D (paralax) của các hạt Gauss. Mô hình thực sự cần thiết ở đây là sự tích hợp phơi sáng 3D (3D exposure integration), bị ràng buộc chặt chẽ bởi cảnh 3D và quỹ đạo camera vật lý, chứ không phải phép nội suy 2D giữa các khung hình nguồn4.

### **Giải pháp Kiến trúc cho Chair: Blur-Aware Gaussian Training (P2)**

Trong điều kiện ánh sáng thực tế và chuyển động nhanh, thuật toán Structure-from-Motion (SfM) như COLMAP thường trích xuất sai pose của camera khi ảnh bị mờ (motion blur)4. Hơn nữa, 3DGS mặc định ép các điểm ảnh sắc nét phải cố tình "phình to" (tăng ma trận hiệp phương sai) hoặc nhiễu loạn để khớp với độ mờ của hình 2D7.  
Dựa trên các nền tảng của DeblurGS4 và Gaussian Splatting on the Move6, quy trình khắc phục cảnh chair sẽ như sau: Mỗi pose máy ảnh được cung cấp không còn là một khoảnh khắc tĩnh (static), mà là **điểm giữa cố định (fixed midpoint)** của một chu kỳ phơi sáng. Mô hình sẽ học cách dự đoán hoặc tối ưu hóa một độ lệch SE(3) nhỏ ở thời điểm bắt đầu/kết thúc cho mỗi khung hình huấn luyện4. Quá trình forward pass sẽ kết xuất ![][image5] mẫu (samples) dọc theo quỹ đạo phơi sáng này, tính trung bình chúng lại và lấy đó làm đầu ra để so sánh (supervise) với hình ảnh bị nhòe thực tế4.  
Các ràng buộc chống học vẹt (anti-overfit constraints) cực kỳ nghiêm ngặt:

> * Pose điểm giữa được cung cấp là bất biến (immutable).  
> * Độ lệch bắt đầu/kết thúc phải có giá trị trung bình bằng 0 (zero-mean) xung quanh điểm giữa đó.  
> * Áp dụng phân phối tiên nghiệm làm mịn (smoothness prior) chung qua các thời điểm chụp lân cận.  
> * Đặt giới hạn (bounds) nhỏ cho phép tịnh tiến/xoay.  
> * Độ lớn của nhòe (blur magnitude) được tham chiếu bởi độ sắc nét đo được của hình ảnh huấn luyện.  
> * Chiến lược "Gaussian Densification Annealing" của DeblurGS là bắt buộc. Trong giai đoạn đầu, khi quỹ đạo camera chưa được ước lượng chính xác, không được phép nhân bản (densify) hạt Gauss, tránh tạo ra các điểm rác sai vị trí4.

Cách tiếp cận này tách rời một cảnh phát quang (radiance field) sắc nét tiềm ẩn ra khỏi sự làm nhòe trong quá trình hình thành ảnh, thay vì yêu cầu bản thân các hạt Gauss phải tự bóp méo để tái tạo một phân phối nhòe thay đổi theo thời gian7.

### **Đánh giá Hàm Mục tiêu và Lựa chọn Checkpoint**

Cơ chế hàm loss "grader" hiện tại đang sử dụng L1 thay cho vị trí của PSNR, và hàm SSIM Gaussian/padded cũ (legacy), sau đó lựa chọn checkpoint thông qua một tập hợp các crop ảnh nhỏ. Mã nguồn hiện đã hỗ trợ tính toán PSNR chuẩn hóa vi phân chính xác (exact differentiable PSNR/50), SSIM có cửa sổ hợp lệ căn chỉnh với công cụ đánh giá (evaluator-aligned valid-window SSIM), và lựa chọn checkpoint cố định bước lặp dựa trên toàn bộ khung hình VGG/SSIM/PSNR.  
Kết quả trên tập holdout bên ngoài bị giới hạn (bounded outer-holdout) như sau:

| 60-pair / 1k arm | Score ngoại vi (external Score) | PSNR | SSIM | LPIPS |
| :---- | :---- | :---- | :---- | :---- |
| DIBR spatial thô (raw spatial DIBR) | **0.658455** | 25.2599 | 0.810170 | **0.340389** |
| L1 cũ \+ SSIM cũ \+ chọn theo crop | 0.657712 | 25.2381 | **0.810473** | 0.342146 |
| PSNR chính xác \+ SSIM chuẩn \+ chọn full-score | 0.657448 | **25.2415** | 0.810110 | 0.342586 |
| loss cũ \+ chỉ chọn full-score | 0.657429 | 25.2191 | 0.810506 | 0.342592 |

Phiên bản sử dụng hàm tính toán chính xác (exact arm) đã đạt đỉnh điểm số (peak score) toàn khung hình nội bộ tại bước lặp 500, trong khi tổn thất trên crop (crop loss) của nó vẫn tiếp tục cải thiện. Điều này xác nhận rằng sự không khớp trong quá trình lựa chọn checkpoint là có thật. Tuy nhiên, mô hình được chọn này lại không tổng quát hóa (generalize) thành công sang một tập 25-frame outer holdout riêng biệt. Do đó, các điều khiển checkpoint chính xác này rất hữu ích cho thử nghiệm A/B sản xuất, nhưng **không được phê duyệt** làm giá trị mặc định cho bản đệ trình. Các tính năng tương thích giai đoạn 2 vòng 2 đã được áp dụng trong Analysis/04\_x3\_dibr\_pilot.py và luồng load ảnh tuần tự được cấu hình để tránh tràn RAM.

## **Cải thiện Chiều sâu Không gian và Bề mặt Vật liệu (Các thay đổi sâu P3)**

Bên cạnh giải quyết bài toán Attention và Blur, việc xử lý tỷ lệ fallback quá cao ở cảnh bonsai yêu cầu can thiệp sâu vào cấu trúc hình học (Geometry) và giao diện quang học phản xạ/tán xạ (Reflective/Translucent Appearance).

### **Hình học: Độ tin cậy và Tinh chỉnh với DepthSplat và PAGaS**

Để cải thiện độ tin cậy của chiều sâu, hướng đi SOTA hiện tại là kết hợp các tiên nghiệm độ sâu đa góc nhìn học được (learned multi-view depth priors) và tinh chỉnh độ sâu bám sát pixel (pixel-aligned depth refinement).

> * **DepthSplat**: Các mô hình 3DGS thường gặp khó khăn tại các vùng thiếu kết cấu (texture-less) hoặc bị phản xạ. DepthSplat giải quyết bằng cách tiêm các đặc trưng độ sâu đơn biến (monocular depth features) \- được trích xuất từ các pre-trained model như Depth Anything \- vào nhánh khớp đặc trưng đa góc nhìn (multi-view feature matching)9. Sự tương tác chéo này (cross-task interactions) giúp tạo ra các độ sâu đa góc nhìn cực kỳ mạnh mẽ, sau đó trực tiếp giải chiếu ngược (unproject) thành các tâm của hạt Gauss trong không gian 3D11. Thậm chí, Gaussian Splatting còn hoạt động ngược lại như một luồng tự giám sát (unsupervised pre-training) để giảm sai số cho mạng ước lượng chiều sâu9.  
> * **PAGaS (Pixel-Aligned 1DoF Gaussian Splatting)**: Sau khi đã có hình học tổng quan tốt, các chi tiết tần số cao (như thớ gỗ, vân lá) vẫn bị mờ. Thay vì để 3DGS tối ưu tự do 59 thông số mỗi hạt (dễ gây overfitting), PAGaS khóa chặt kích thước, góc quay và độ mờ (![][image6]) của hạt Gauss14. Vị trí không gian ![][image1] được gán cứng theo tia chiếu ngược của pixel ảnh. Tham số duy nhất (1-DoF) được tối ưu hóa chính là **chiều sâu ![][image7]** chạy dọc theo tia nhìn đó, với độ sâu Euclidean ![][image8]15. Việc này đảm bảo hạt Gauss không bị phình to hay lấn át lẫn nhau, bảo toàn tuyệt đối chi tiết hình học tại các bề mặt mảnh mai14.

### **Quang học: Phản xạ Gương và Tán xạ Dưới bề mặt (SpecTRe-GS & SSS-GS)**

Sau khi khắc phục được sai số độ bao phủ (coverage error), mô hình cần tái tạo độ bóng của mặt bàn (chair) và độ thấu quang của lá cây (bonsai). Các mô hình SH (Spherical Harmonics) cơ bản là bộ lọc thông thấp (low-pass filter) nên thất bại trước ánh sáng phản chiếu gắt (specular).

> * **SpecTRe-GS**: Hệ thống này phân tách ánh sáng từ các bề mặt láng bóng (highly specular) và bề mặt thô (rough)17. Thay vì lưu màu sắc chết, nó tích hợp một bộ dò tia (ray tracer) hiệu suất cao trực tiếp vào framework 3DGS. Nó bắn các tia phụ (secondary rays) từ bề mặt để lấy dữ liệu từ các vật thể lân cận phản chiếu vào17. Kết hợp với kỹ thuật đổ bóng trễ (deferred shading) qua G-buffers, SpecTRe-GS có thể tính toán các phương trình kết xuất vật lý phức tạp, tạo ra bóng đổ và phản xạ chân thực18.  
> * **SSS-GS (Subsurface Scattering cho 3DGS)**: Lá cây bonsai cho phép ánh sáng xuyên qua và tán xạ bên trong. SSS-GS phân rã cảnh thành hai phần: bề mặt rõ ràng (explicit surface) mô tả bởi GS với BRDF, và một biểu diễn thể tích ẩn (implicit volumetric representation) mô tả sự tán xạ20. Các phương pháp như RT-Splatting tách rời độ chiếm dụng hình học (geometric occupancy) khỏi độ mờ quang học (optical opacity), giúp hệ thống hiển thị chính xác các đối tượng bán trong suốt mà không làm hỏng tính toán che khuất (occlusion) phía sau23.

Bất kỳ mô hình chiều sâu (monocular/depth) pre-trained nào được áp dụng (như DepthSplat) đều phải trải qua quá trình kiểm tra tuân thủ bộ quy tắc về nguồn gốc và dữ liệu bên ngoài của cuộc thi.

## **Khuyến nghị Thực thi và Quản lý Tài nguyên GPU**

Mức tiêu thụ VRAM đỉnh điểm được ghi nhận trên thiết bị RTX 4070 Ti SUPER:

| Khối lượng công việc (Workload) | VRAM ghi nhận (Observed VRAM) |
| :---- | :---- |
| Huấn luyện splatfacto chống răng cưa 10k (10k antialiased) | khoảng 4.0 GB |
| Áp dụng DIBR/refiner phân giải đầy đủ (full-resolution) | khoảng 7.1 GB |
| Refiner metric chính xác, 256 crops, batch 4, base 32 | khoảng 7.3 GB |

Card đồ họa 16 GB hiện tại là **hoàn toàn đủ** để đáp ứng các mô hình thí điểm đề xuất. Đối với tính năng chú ý nguồn K=5 (K=5 evidence), cần thiết lập batch size xuống 2 và base ở mức 32–48; lưu giữ hình ảnh trên RAM CPU và đẩy sang tính toán thông qua gsplat bất đồng bộ. Một GPU 24–48 GB sẽ mang lại tốc độ và xử lý các batch evidence lớn hơn, nhưng không bắt buộc để kiểm định tính khả thi của kỹ thuật.  
Trong môi trường cục bộ (local environment), cần một trình biên dịch CUDA thực thụ để biên dịch JIT cho gsplat. Hãy khởi chạy thông qua Conda để đảm bảo \<env\>/bin/nvcc nằm trên biến môi trường PATH:

Bash  
conda run \--no-capture-output \-n airace python ...

Tệp environment.yml của kho lưu trữ đã cố định (pin) phiên bản CUDA 12.1 compiler/toolkit. Yêu cầu sản xuất cục bộ còn lại là tệp thực thi colmap cho mảng Dense MVS; còn đối với các thử nghiệm nhẹ (lightweight pilots) có thể tái sử dụng hình học thưa (sparse geometry) được cung cấp sẵn.

## **Trình tự Triển khai Cập nhật (Adoption Order)**

Để quản lý rủi ro và ngăn ngừa regression trên điểm số gốc, luồng triển khai cần được tuân thủ nghiêm ngặt:

> 1. **Phục hồi/tái thiết (Restore/rebuild)** neo kiểm định 30k dense/SSS holdout (lớp validation layer). Việc kiểm định phải qua 2 lớp: hình ảnh cố định nội bộ cho checkpoint và 25-frame match-test holdout để quyết định áp dụng.  
> 2. **Triển khai P1 (Per-pixel source attention)** với K=5 cho cảnh bonsai; đặt ngưỡng thông qua (gate) là tăng tối thiểu \+0.003 điểm global và \+0.008 trên các phân khúc fallback cao/giai đoạn đầu.  
> 3. **Triển khai P2 (Chair fixed-midpoint exposure integration)** với M=5; chỉ xác nhận chạy với cấu hình M=9 sau khi mô hình đã qua được cổng đánh giá dương tính.  
> 4. **Thử nghiệm P3 (Độ tin cậy chiều sâu/tinh chỉnh)** qua PAGaS và DepthSplat trên cảnh bonsai.  
> 5. **Thử nghiệm P3 (Giao diện phản quang)** qua SpecTRe-GS và SSS-GS chỉ sau khi sai số che phủ (coverage error) đã được giảm thiểu hoàn toàn từ các bước trước.  
> 6. **Chạy lại (Re-run)** tùy chọn hàm loss chính xác (exact loss) / bộ chọn checkpoint (checkpoint selector) trên neo sản xuất; tuyệt đối không suy diễn khả năng áp dụng từ các pilot thưa thớt (sparse pilot).

Tóm lại, con đường thực tế nhất để vươn tới cột mốc 77 không nằm ở việc quét một loạt thông số toàn hạm đội (fleet-wide parameter sweep). Nó yêu cầu **hai sự thay đổi mô hình mang tính đặc thù cho từng cảnh**: cơ chế chọn bằng chứng bề mặt học được (learned surface-wise evidence selection) cho bonsai, và mô phỏng vật lý sự hình thành độ nhòe (physically constrained blur formation) cho chair. Những cải tiến sâu sắc này đòi hỏi kỹ sư Machine Learning phải thoát khỏi các mô hình 2D quen thuộc để làm chủ các ràng buộc không gian đa chiều (SE3, Ray Tracing, 1DoF Gaussian) của 3D Gaussian Splatting.

#### **Nguồn trích dẫn**

> 1. IBRNet: Learning Multi-View Image-Based Rendering \- CVF Open Access, [https://openaccess.thecvf.com/content/CVPR2021/papers/Wang\_IBRNet\_Learning\_Multi-View\_Image-Based\_Rendering\_CVPR\_2021\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_IBRNet_Learning_Multi-View_Image-Based_Rendering_CVPR_2021_paper.pdf)  
> 2. IBRNet: Learning Multi-View Image-Based Rendering \- Semantic Scholar, [https://www.semanticscholar.org/paper/IBRNet%3A-Learning-Multi-View-Image-Based-Rendering-Wang-Wang/7cbc3dd0280b8c4551ac934af42dc227d43754f7](https://www.semanticscholar.org/paper/IBRNet%3A-Learning-Multi-View-Image-Based-Rendering-Wang-Wang/7cbc3dd0280b8c4551ac934af42dc227d43754f7)  
> 3. \[2102.13090\] IBRNet: Learning Multi-View Image-Based Rendering \- arXiv, [https://arxiv.org/abs/2102.13090](https://arxiv.org/abs/2102.13090)  
> 4. DeblurGS: Gaussian Splatting for Camera Motion Blur \- arXiv, [https://arxiv.org/html/2404.11358v2](https://arxiv.org/html/2404.11358v2)  
> 5. \[2404.11358\] DeblurGS: Gaussian Splatting for Camera Motion Blur \- arXiv, [https://arxiv.org/abs/2404.11358](https://arxiv.org/abs/2404.11358)  
> 6. SpectacularAI/3dgs-deblur: \[ECCV2024\] Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion \- GitHub, [https://github.com/SpectacularAI/3dgs-deblur](https://github.com/SpectacularAI/3dgs-deblur)  
> 7. BARD-GS: Blur-Aware Reconstruction of Dynamic Scenes via Gaussian Splatting \- arXiv, [https://arxiv.org/html/2503.15835v1](https://arxiv.org/html/2503.15835v1)  
> 8. DeblurGS: Gaussian Splatting for Camera Motion Blur \- GitHub, [https://github.com/taekkii/deblurgs](https://github.com/taekkii/deblurgs)  
> 9. DepthSplat: Connecting Gaussian Splatting and Depth \- OpenReview, [https://openreview.net/forum?id=IcPkW3QNW2](https://openreview.net/forum?id=IcPkW3QNW2)  
> 10. CVPR Poster DepthSplat: Connecting Gaussian Splatting and Depth, [https://cvpr.thecvf.com/virtual/2025/poster/32696](https://cvpr.thecvf.com/virtual/2025/poster/32696)  
> 11. DepthSplat: Connecting Gaussian Splatting and Depth \- CVF Open Access, [https://openaccess.thecvf.com/content/CVPR2025/papers/Xu\_DepthSplat\_Connecting\_Gaussian\_Splatting\_and\_Depth\_CVPR\_2025\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_DepthSplat_Connecting_Gaussian_Splatting_and_Depth_CVPR_2025_paper.pdf)  
> 12. (PDF) DepthSplat: Connecting Gaussian Splatting and Depth \- ResearchGate, [https://www.researchgate.net/publication/385010080\_DepthSplat\_Connecting\_Gaussian\_Splatting\_and\_Depth](https://www.researchgate.net/publication/385010080_DepthSplat_Connecting_Gaussian_Splatting_and_Depth)  
> 13. \[2410.13862\] DepthSplat: Connecting Gaussian Splatting and Depth \- arXiv, [https://arxiv.org/abs/2410.13862](https://arxiv.org/abs/2410.13862)  
> 14. PAGaS: Pixel-Aligned 1DoF Gaussian Splatting for Depth Refinement \- arXiv, [https://arxiv.org/html/2604.22129v1](https://arxiv.org/html/2604.22129v1)  
> 15. \[Literature Review\] PAGaS: Pixel-Aligned 1DoF Gaussian Splatting for Depth Refinement, [https://www.themoonlight.io/en/review/pagas-pixel-aligned-1dof-gaussian-splatting-for-depth-refinement](https://www.themoonlight.io/en/review/pagas-pixel-aligned-1dof-gaussian-splatting-for-depth-refinement)  
> 16. PAGaS: Pixel-Aligned 1DoF Gaussian Splatting for Depth Refinement \- CVF Open Access, [https://openaccess.thecvf.com/content/CVPR2026W/3DMV/papers/Recasens\_PAGaS\_Pixel-Aligned\_1DoF\_Gaussian\_Splatting\_for\_Depth\_Refinement\_CVPRW\_2026\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2026W/3DMV/papers/Recasens_PAGaS_Pixel-Aligned_1DoF_Gaussian_Splatting_for_Depth_Refinement_CVPRW_2026_paper.pdf)  
> 17. CVPR Poster SpecTRe-GS: Modeling Highly Specular Surfaces with Reflected Nearby Objects by Tracing Rays in 3D Gaussian Splatting, [https://cvpr.thecvf.com/virtual/2025/poster/33748](https://cvpr.thecvf.com/virtual/2025/poster/33748)  
> 18. SpecTRe-GS: Modeling Highly Specular Surfaces with Reflected Nearby Objects by Tracing Rays in 3D Gaussian Splatting \- CVF Open Access, [https://openaccess.thecvf.com/content/CVPR2025/papers/Tang\_SpecTRe-GS\_Modeling\_Highly\_Specular\_Surfaces\_with\_Reflected\_Nearby\_Objects\_by\_CVPR\_2025\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_SpecTRe-GS_Modeling_Highly_Specular_Surfaces_with_Reflected_Nearby_Objects_by_CVPR_2025_paper.pdf)  
> 19. SpecTRe-GS: Modeling Highly Specular Surfaces with Reflected Nearby Objects by Tracing Rays in 3D Gaussian Splatting \- CVPR 2025 Open Access Repository, [https://openaccess.thecvf.com/content/CVPR2025/html/Tang\_SpecTRe-GS\_Modeling\_Highly\_Specular\_Surfaces\_with\_Reflected\_Nearby\_Objects\_by\_CVPR\_2025\_paper.html](https://openaccess.thecvf.com/content/CVPR2025/html/Tang_SpecTRe-GS_Modeling_Highly_Specular_Surfaces_with_Reflected_Nearby_Objects_by_CVPR_2025_paper.html)  
> 20. Subsurface Scattering for Gaussian Splatting \- OpenReview, [https://openreview.net/forum?id=2vMvh5XP0P](https://openreview.net/forum?id=2vMvh5XP0P)  
> 21. \[2408.12282\] Subsurface Scattering for 3D Gaussian Splatting \- arXiv, [https://arxiv.org/abs/2408.12282](https://arxiv.org/abs/2408.12282)  
> 22. Subsurface Scattering for 3D Gaussian Splatting \- arXiv, [https://arxiv.org/html/2408.12282v2](https://arxiv.org/html/2408.12282v2)  
> 23. RT-Splatting: Joint Reflection-Transmission Modeling with Gaussian Splatting \- arXiv, [https://arxiv.org/html/2605.18263v1](https://arxiv.org/html/2605.18263v1)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAbCAYAAABIpm7EAAAA1ElEQVR4XmNgGAXDAswG4gVoYrlAfBJNDAxYgPgLEE9EEz8BxNvRxMDAHIj/A3EokhgXEP8C4nokMTgoY4BokEASc4CKuSGJwQHJGrYA8W00sVog/gfE/GjiDExA/J4BM4R2AvEVKNsSWUKfAWJ1M5IYyNRPDJCgZgbifUhyDDkMEA1roXweIF4PxD+BuAmIfYG4AioHBquA+A0QH2eARNJ+IHYG4iQGiL+2MkAMgYNnQLwCWQAfUGaAOCcbXQIXiGOAaNBBl8AFCoD4DBAzoksMVwAAhOgqRM26O90AAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAZCAYAAADuWXTMAAAA40lEQVR4Xu2SqwoCURCGBzQatHp/AJvFC/gCRptJsRjEB1DB5KNYDILFLiaxiihoMZkEs3j5Z+ccWYZdvCSDH3xl/p05u3uG6M9XHOEdnuHGuIV7k91MziZMz5O6CU4wqTImAnskzxRV5jAiCecwoDLLGFZ1kQnDA8mAvsos3NjRRUsJXuEF5lTGxGBZF90MSE7fwZDKXhKEC5IBTZW9RRdOSQZ9BH/TkrxfOQNbumjJwjWM6sDQgG1dZFIkm8XT/ZjBii7yHa9gDcaVaViAQ5KfmHc6XExI7ta9w356re+f3+YBG9UyZxfKwaYAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAbCAYAAACnZAX6AAAAxElEQVR4XmNgGAUjBngD8X4oPgvEjUDMhCTPiMQGgxogfgrEmlC+CBC/AuJaKL8MiD2hbDAIBeL/DBCbkMFsIH4NxMxAvBeI2WASIIEHQPwQJoAEQLaADIsA4snIEmRpMoNKzEQWhIICBojcVSAWQ5YIh0okIwtCQQ4DRC4PXcIKKhGALgEEdQwQOZBrUAAoHs4D8SwkMXEGiB9WMyAMDAFiBSQ1DLJAvAGIjwDxDiBeDsTWULlJQHwLiBcxYIncUTB4AQBczSfTPPQZ5QAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAABDUlEQVR4XmNgGAXDFzgA8XUg/gLE/4H4AxDfBGIfJDWtUDkQ/grEs5HksIIdDBDFRugSQGALxMeB2AWIGdHkMAArEH8G4lcMqIqZgLgaiJuAmAVJHC+wYYC4aimSmAwQLwdiZyQxokAjA8SwOCg/EIiPArEoXAUJ4BgDxDAlIJ4MxD8ZIN7mR1ZEDABp+APE74B4IxCbAfEMBojhpUjqiAIgL4E0gmKTFyqmBsT/gPgRAwkBDwLTGSCGuaGJg1wJEo9CE8cL7gLxDyDmRBO3Y4AYdh5NHCfQZIBo2IcuAQW3GCDy3ugSyMCBAZKFPjFAFINi7ioQu0Pl2aDyoHADyYNiF+RCZqj8KBgFdAUA1cE4UD3uB1gAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADwAAAAaCAYAAADrCT9ZAAACI0lEQVR4Xu2WP0hVURzHf2maTpaghCJEizhkgemYDgbZ0BDV0JAGjiniFipIKjikgoMgCE0mCA4SgRA05FCJRSEuYmqoNTmoIIKIfb/vd6/v3PN83ifBuw85H/gM9/c7973zO//uEXE4HI5ouQebYBnMgxVwELYbbc4VPfDIckV0AEKpgotwW/TF1WA6gcei7Q7hMpwIptNCN/wNN+BP2AcLzAapMA5/iBZyycr5XIUDogW/sXLppEt0Sf8XC7BDtJhyK+fTCjtF2zyxcumEfWiyg2eBm35S4su1IZiO8QBehx9FV0FhMJ1WODGv4RSchXPwfqBFCJy5ZtH9zIJfBNNSBJ/CfLgPvwbTSeF730W3SqrW8cUQXsJPEt+39fAA3j1uEcI7eA1eFi14KJDVAcgS/UHmeUpGSSm8YsV4iHHAQskRnQWfLThtPLNIf0/3ixZ8J57OGLi02bcSO2HDzo8Yz9wP/EwRzvgzIzcPd0UHKSo4u3/hsBXn2cKCb1vxBHrhQ+P5LdyDF2ALvOjFuYR4WL33nlOBe5gDyIFK1drYm8mpEy3sgxXn/zDOATmVL6Iz6ePfYp7Dm0acg8J4mxGLgmLRW5XZ51y4Az8bsROphL9EDySfRtHC+HE3GfXiN6x4FPDe/Apmi65AXoa41W6ZjUxq4JroEmURPKgeeblq+E3it60xuOm1o+ui3+woYaHcikvwD5wRnTyHw+FwOBwZxj8FwHu5OPEvXQAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADMAAAAaCAYAAAAaAmTUAAABSUlEQVR4Xu2WsS8EQRTGP4RWI6eSKCm0CvpDNApCrxVRqZD4BxSuQqdRaJQa0UgoCI1GFEhcQSQKgij4njeJuZeL7Bbmsrxf8kt23rebzNvJ7A7gOM5fptsWikYr7aEr9NFkhaKPPtBDekOfauPisov/3Mwo3Q+e0GXaHOVN0XVqcjWzQG9pbxh30Du6GMbzdCRc12OInuXwmLZ/PZmNzM1M0A/oysRs0HvaQvdoW22clEzNyESv6LWpC7Iq0uQUrZgsNdLMsy1a+qETXrMBmYNm57RkstRIMy+2aJmETnjaBmQGms3aoA5l6D7I6hHy75lXW7QMQCc8ZgOyBM1k9RqNNPNmixb59J7S9ajWCd0j2/hudByNPRvJB+gderz5kS66Qw+gb2CLDoZslV7QTaT/z8jv4ZJWoS9VlPOZ1Iaj+xzHcRzn1/kELKtT29rqByEAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAZCAYAAADnstS2AAAAvUlEQVR4XmNgGAXEAScgfgzE/4H4BJocViDKAFHciS6BDUQxQBQ7o0tgA4uB+CsQs6NLsAFxLRDvBuJNQNwNxK+BeBuyIhDgAuJDQLyTAWHKbAaIE/JhimBgMhD/AWIFJLEyBohiDSQxhiCoYDmSGAcQfwfiWUhiYBDAAFHshiQG8j1ILASIbYA4CyYhzgAxJRLKlwXiKwwQxXJAPB+I1aFyYOAOxEeAeB0QLwJibSDeA8T7gbgKSd0ooAMAAAv0IqpWzryXAAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,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>