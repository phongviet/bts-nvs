<div class="max-w-none text-sm leading-relaxed text-foreground [&amp;_h1]:mb-3 [&amp;_h1]:mt-6 [&amp;_h1]:text-2xl [&amp;_h1]:font-extrabold [&amp;_h2]:mb-2 [&amp;_h2]:mt-6 [&amp;_h2]:text-xl [&amp;_h2]:font-bold [&amp;_h3]:mb-2 [&amp;_h3]:mt-4 [&amp;_h3]:text-lg [&amp;_h3]:font-bold [&amp;_p]:my-3 [&amp;_p]:text-foreground/90 [&amp;_ul]:my-3 [&amp;_ul]:list-disc [&amp;_ul]:pl-6 [&amp;_ol]:my-3 [&amp;_ol]:list-decimal [&amp;_ol]:pl-6 [&amp;_li]:my-1 [&amp;_li]:text-foreground/90 [&amp;_a]:font-medium [&amp;_a]:text-primary [&amp;_a]:underline [&amp;_a]:underline-offset-2 [&amp;_strong]:font-semibold [&amp;_strong]:text-foreground [&amp;_code]:rounded [&amp;_code]:bg-muted [&amp;_code]:px-1.5 [&amp;_code]:py-0.5 [&amp;_code]:font-mono [&amp;_code]:text-[0.85em] [&amp;_pre]:my-4 [&amp;_pre]:overflow-x-auto [&amp;_pre]:rounded-lg [&amp;_pre]:bg-slate-900 [&amp;_pre]:p-4 [&amp;_pre]:text-slate-50 [&amp;_pre_code]:bg-transparent [&amp;_pre_code]:p-0 [&amp;_pre_code]:text-slate-50 [&amp;_blockquote]:my-4 [&amp;_blockquote]:border-l-4 [&amp;_blockquote]:border-primary/30 [&amp;_blockquote]:pl-4 [&amp;_blockquote]:italic [&amp;_blockquote]:text-muted-foreground [&amp;_table]:my-4 [&amp;_table]:w-full [&amp;_table]:border-collapse [&amp;_th]:border [&amp;_th]:border-border [&amp;_th]:bg-muted [&amp;_th]:px-3 [&amp;_th]:py-2 [&amp;_th]:text-left [&amp;_td]:border [&amp;_td]:border-border [&amp;_td]:px-3 [&amp;_td]:py-2 [&amp;_hr]:my-6 [&amp;_hr]:border-border [&amp;_img]:my-4 [&amp;_img]:rounded-lg [&amp;_.katex-display]:my-4 [&amp;_.katex-display]:overflow-x-auto [&amp;_.katex-display]:overflow-y-hidden"><p>Bài toán yêu cầu thí sinh xây dựng hệ thống AI có khả năng tái dựng cấu trúc 3D ngầm định của trạm BTS từ tập ảnh drone, và sinh ảnh RGB tại các góc nhìn chưa từng được chụp. Đây là hướng tiếp cận hiện đại cho việc xây dựng Digital Twin - bản sao số 3D có độ chính xác cao của hạ tầng viễn thông - phục vụ giám sát, kiểm tra, bảo trì và quy hoạch lắp đặt thiết bị. Mỗi scene gồm 100-300 ảnh RGB kèm thông số camera và pose tương ứng; thí sinh cần sinh ảnh tại 20-50 góc nhìn mục tiêu, đảm bảo đúng về hình học, vị trí thiết bị và chất lượng hình ảnh chân thực.</p></div>

<div class="max-w-none text-sm leading-relaxed text-foreground [&amp;_h1]:mb-3 [&amp;_h1]:mt-6 [&amp;_h1]:text-2xl [&amp;_h1]:font-extrabold [&amp;_h2]:mb-2 [&amp;_h2]:mt-6 [&amp;_h2]:text-xl [&amp;_h2]:font-bold [&amp;_h3]:mb-2 [&amp;_h3]:mt-4 [&amp;_h3]:text-lg [&amp;_h3]:font-bold [&amp;_p]:my-3 [&amp;_p]:text-foreground/90 [&amp;_ul]:my-3 [&amp;_ul]:list-disc [&amp;_ul]:pl-6 [&amp;_ol]:my-3 [&amp;_ol]:list-decimal [&amp;_ol]:pl-6 [&amp;_li]:my-1 [&amp;_li]:text-foreground/90 [&amp;_a]:font-medium [&amp;_a]:text-primary [&amp;_a]:underline [&amp;_a]:underline-offset-2 [&amp;_strong]:font-semibold [&amp;_strong]:text-foreground [&amp;_code]:rounded [&amp;_code]:bg-muted [&amp;_code]:px-1.5 [&amp;_code]:py-0.5 [&amp;_code]:font-mono [&amp;_code]:text-[0.85em] [&amp;_pre]:my-4 [&amp;_pre]:overflow-x-auto [&amp;_pre]:rounded-lg [&amp;_pre]:bg-slate-900 [&amp;_pre]:p-4 [&amp;_pre]:text-slate-50 [&amp;_pre_code]:bg-transparent [&amp;_pre_code]:p-0 [&amp;_pre_code]:text-slate-50 [&amp;_blockquote]:my-4 [&amp;_blockquote]:border-l-4 [&amp;_blockquote]:border-primary/30 [&amp;_blockquote]:pl-4 [&amp;_blockquote]:italic [&amp;_blockquote]:text-muted-foreground [&amp;_table]:my-4 [&amp;_table]:w-full [&amp;_table]:border-collapse [&amp;_th]:border [&amp;_th]:border-border [&amp;_th]:bg-muted [&amp;_th]:px-3 [&amp;_th]:py-2 [&amp;_th]:text-left [&amp;_td]:border [&amp;_td]:border-border [&amp;_td]:px-3 [&amp;_td]:py-2 [&amp;_hr]:my-6 [&amp;_hr]:border-border [&amp;_img]:my-4 [&amp;_img]:rounded-lg [&amp;_.katex-display]:my-4 [&amp;_.katex-display]:overflow-x-auto [&amp;_.katex-display]:overflow-y-hidden"><h1>1. Tổng quan bài toán</h1>
<p>Mục tiêu của bài toán là xây dựng mô hình AI có khả năng tái dựng cấu trúc không gian 3D của một scene từ tập ảnh đa góc nhìn và sinh ra ảnh tại các góc nhìn mới chưa từng xuất hiện trong dữ liệu đầu vào.</p>
<p>Dữ liệu có thể được thu thập từ:</p>
<ul>
<li>Drone bay quanh đối tượng,</li>
<li>Camera cầm tay (hand-held camera).</li>
</ul>
<p>Đối tượng trong scene có thể là:</p>
<ul>
<li>Trạm BTS</li>
<li>Công trình hạ tầng</li>
<li>Các đối tượng thực tế khác</li>
</ul>
<p>Bài toán thuộc các lĩnh vực:</p>
<ul>
<li>Computer Vision</li>
<li>3D Vision</li>
<li>Neural Rendering</li>
<li>Novel View Synthesis</li>
<li>Digital Twin</li>
</ul>
<hr>
<h1>2. Cấu trúc dữ liệu</h1>
<p>Mỗi scene dữ liệu có cấu trúc như sau:</p>
<pre><code class="language-text">

├── train/
│   ├── images/          : Ảnh training
│   ├── sparse/0/        : Sparse reconstruction từ COLMAP
│   │                       ├── cameras.bin
│   │                       ├── images.bin
│   │                       └── points3D.bin
└── test/
    └── test_poses.csv   : Camera poses cho test images
</code></pre>
<hr>
<h1>3. Thông tin dữ liệu</h1>
<ul>
<li>Train images: ~80%</li>
<li>Test images: ~20%</li>
<li>Camera poses và sparse reconstruction đã được dựng sẵn bằng COLMAP và cung cấp cho thí sinh</li>
</ul>
<hr>
<h1>4. Format test_poses.csv</h1>
<pre><code class="language-text">image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height
</code></pre>
<p>Trong đó:</p>
<ul>
<li><code>image_name</code>: tên ảnh đầu ra cần sinh</li>
<li><code>qw, qx, qy, qz</code>: quaternion rotation theo format COLMAP</li>
<li><code>tx, ty, tz</code>: camera translation</li>
<li><code>fx, fy</code>: focal length</li>
<li><code>cx, cy</code>: principal point</li>
<li><code>width, height</code>: kích thước ảnh cần sinh</li>
</ul>
<hr>
<h1>5. Đầu vào bài toán</h1>
<p>Đầu vào bao gồm:</p>
<ul>
<li>tập ảnh train đa góc nhìn</li>
<li>camera intrinsics</li>
<li>camera poses</li>
<li>sparse reconstruction từ COLMAP</li>
<li>danh sách test poses</li>
</ul>
<hr>
<h1>6. Đầu ra bài toán</h1>
<p>Thí sinh cần sinh:</p>
<ul>
<li>ảnh RGB tương ứng với toàn bộ test poses được cung cấp</li>
</ul>
<p>Ảnh đầu ra cần:</p>
<ul>
<li>đúng cấu trúc hình học</li>
<li>đúng vị trí các vật thể</li>
<li>đảm bảo chất lượng hình ảnh chân thực và nhất quán</li>
</ul>
<hr>
<h1>7. Format submission</h1>
<p>Submission là file ZIP chứa toàn bộ ảnh kết quả:</p>
<pre><code class="language-text">submission.zip
├── scene_001/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── scene_002/
│   ├── 0001.png
│   └── ...
└── ...
</code></pre>
<p>Yêu cầu:</p>
<ul>
<li>Đúng số lượng và tên scene</li>
<li>Đúng tên file ảnh</li>
<li>Đúng kích thước ảnh</li>
<li>Đúng số lượng ảnh mỗi scene</li>
</ul>
<hr>
<h1>8. Metrics đánh giá</h1>
<p>Kết quả được đánh giá bằng cách so sánh ảnh sinh ra với ảnh ground-truth bằng ba metrics:</p>
<hr>
<h2>8.1 LPIPS</h2>
<p>Đánh giá độ tương đồng cảm quan giữa hai ảnh bằng đặc trưng deep learning</p>
<ul>
<li>Giá trị càng thấp càng tốt.</li>
</ul>
<p>Tham khảo:</p>
<pre><code class="language-text">Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang.
"The Unreasonable Effectiveness of Deep Features as a Perceptual Metric."
CVPR 2018.
https://arxiv.org/abs/1801.03924
</code></pre>
<hr>
<h2>8.2 SSIM</h2>
<p>Đánh giá độ tương đồng về cấu trúc hình ảnh</p>
<ul>
<li>Giá trị càng cao càng tốt.</li>
</ul>
<p>Tham khảo:</p>
<pre><code class="language-text">Zhou Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli.
"Image quality assessment: from error visibility to structural similarity."
IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, April 2004.
doi: 10.1109/TIP.2003.819861
</code></pre>
<hr>
<h2>8.3 PSNR</h2>
<p>Đánh giá sai số mức pixel giữa ảnh dự đoán và ground-truth</p>
<ul>
<li>Giá trị càng cao càng tốt.</li>
</ul>
<p>Tham khảo:</p>
<pre><code class="language-text">Zhou Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli.
"Image quality assessment: from error visibility to structural similarity."
IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, April 2004.
doi: 10.1109/TIP.2003.819861
</code></pre>
<p>Để kết hợp với các metrics khác, giá trị PSNR sẽ được chuẩn hóa về khoảng [0,1] theo công thức:</p>
<pre><code class="language-python">psnr_norm = torch.clamp(psnr_val / psnr_max, 0.0, 1.0)
</code></pre>
<p>Trong đó:</p>
<ul>
<li><code>PSNR_max</code> là ngưỡng PSNR tối đa được lựa chọn trước</li>
<li><code>clamp</code> dùng để giới hạn giá trị trong khoảng từ 0 đến 1</li>
</ul>
<hr>
<h2>8.4. Công thức tính điểm cuối cùng</h2>
<span class="katex-display"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mi>S</mi><mi>c</mi><mi>o</mi><mi>r</mi><mi>e</mi><mo>=</mo><mn>0.4</mn><mo>×</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>L</mi><mi>P</mi><mi>I</mi><mi>P</mi><mi>S</mi><mo stretchy="false">)</mo><mo>+</mo><mn>0.3</mn><mo>×</mo><mi>S</mi><mi>S</mi><mi>I</mi><mi>M</mi><mo>+</mo><mn>0.3</mn><mo>×</mo><mi>P</mi><mi>S</mi><mi>N</mi><msub><mi>R</mi><mrow><mi>n</mi><mi>o</mi><mi>r</mi><mi>m</mi></mrow></msub></mrow><annotation encoding="application/x-tex">Score =
0.4 \times (1-LPIPS)
+
0.3 \times SSIM
+
0.3 \times PSNR_{norm}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.6833em;"></span><span class="mord mathnormal" style="margin-right: 0.0576em;">S</span><span class="mord mathnormal" style="margin-right: 0.0278em;">cor</span><span class="mord mathnormal">e</span><span class="mspace" style="margin-right: 0.2778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.2778em;"></span></span><span class="base"><span class="strut" style="height: 0.7278em; vertical-align: -0.0833em;"></span><span class="mord">0.4</span><span class="mspace" style="margin-right: 0.2222em;"></span><span class="mbin">×</span><span class="mspace" style="margin-right: 0.2222em;"></span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mopen">(</span><span class="mord">1</span><span class="mspace" style="margin-right: 0.2222em;"></span><span class="mbin">−</span><span class="mspace" style="margin-right: 0.2222em;"></span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal">L</span><span class="mord mathnormal" style="margin-right: 0.1389em;">P</span><span class="mord mathnormal" style="margin-right: 0.0785em;">I</span><span class="mord mathnormal" style="margin-right: 0.1389em;">P</span><span class="mord mathnormal" style="margin-right: 0.0576em;">S</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.2222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right: 0.2222em;"></span></span><span class="base"><span class="strut" style="height: 0.7278em; vertical-align: -0.0833em;"></span><span class="mord">0.3</span><span class="mspace" style="margin-right: 0.2222em;"></span><span class="mbin">×</span><span class="mspace" style="margin-right: 0.2222em;"></span></span><span class="base"><span class="strut" style="height: 0.7667em; vertical-align: -0.0833em;"></span><span class="mord mathnormal" style="margin-right: 0.0576em;">S</span><span class="mord mathnormal" style="margin-right: 0.0576em;">S</span><span class="mord mathnormal" style="margin-right: 0.0785em;">I</span><span class="mord mathnormal" style="margin-right: 0.109em;">M</span><span class="mspace" style="margin-right: 0.2222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right: 0.2222em;"></span></span><span class="base"><span class="strut" style="height: 0.7278em; vertical-align: -0.0833em;"></span><span class="mord">0.3</span><span class="mspace" style="margin-right: 0.2222em;"></span><span class="mbin">×</span><span class="mspace" style="margin-right: 0.2222em;"></span></span><span class="base"><span class="strut" style="height: 0.8333em; vertical-align: -0.15em;"></span><span class="mord mathnormal" style="margin-right: 0.1389em;">P</span><span class="mord mathnormal" style="margin-right: 0.0576em;">S</span><span class="mord mathnormal" style="margin-right: 0.109em;">N</span><span class="mord"><span class="mord mathnormal" style="margin-right: 0.0077em;">R</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.1514em;"><span style="top: -2.55em; margin-left: -0.0077em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">n</span><span class="mord mathnormal mtight" style="margin-right: 0.0278em;">or</span><span class="mord mathnormal mtight">m</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span></span></span></span></span></span></span></span></span></span></span>
<p>Điểm trên bảng xếp hạng là điểm trung bình của toàn bộ các scene, nếu thiếu scene hoặc thừa scene so với groundtruth, kết quả sẽ không được tính.</p>
<h1>9. Hình thức thi</h1>
<p>Dữ liệu và scene hoàn toàn mới được cung cấp cho mỗi vòng thi, cách thức tính điểm sẽ được giữ nguyên.</p>
<hr>
<h1>10. Quy định chống gian lận và đảm bảo tính công bằng</h1>
<p>Để đảm bảo cuộc thi đánh giá đúng năng lực xây dựng mô hình AI của thí sinh, Ban Tổ Chức áp dụng các quy định sau:</p>
<h2>10.1. Cấm sử dụng dữ liệu ngoài</h2>
<p>Thí sinh chỉ được phép sử dụng dữ liệu do Ban Tổ Chức cung cấp trong từng vòng thi.</p>
<p>Nghiêm cấm:</p>
<ul>
<li>Sử dụng ảnh, video hoặc dữ liệu 3D bên ngoài có chứa cùng đối tượng hoặc cùng scene của bộ dữ liệu thi</li>
<li>Thu thập bổ sung dữ liệu thực địa hoặc từ Internet liên quan trực tiếp đến các scene được cung cấp</li>
<li>Sử dụng bất kỳ nguồn dữ liệu nào nhằm tái tạo hoặc suy luận ground-truth của tập test</li>
</ul>
<h2>10.2. Cấm truy xuất hoặc suy đoán dữ liệu kiểm thử</h2>
<p>Nghiêm cấm mọi hành vi nhằm:</p>
<ul>
<li>Truy cập trái phép vào dữ liệu ground-truth</li>
<li>Khai thác lỗ hổng hệ thống để thu thập thông tin về ảnh kiểm thử</li>
</ul>
<h2>10.3. Yêu cầu khả năng tái lập kết quả</h2>
<p>Ban Tổ Chức có quyền yêu cầu các đội đạt thứ hạng cao cung cấp:</p>
<ul>
<li>Mã nguồn huấn luyện và suy luận</li>
<li>File cấu hình (config)</li>
<li>Danh sách thư viện và phiên bản sử dụng</li>
<li>Checkpoint mô hình</li>
<li>Nhật ký huấn luyện (training logs)</li>
</ul>
<p>Đội thi phải chứng minh rằng kết quả nộp bài có thể được tái tạo từ pipeline đã công bố.</p>
<h2>10.4. Cấm chỉnh sửa thủ công ảnh đầu ra</h2>
<p>Toàn bộ ảnh kết quả phải được sinh tự động bởi thuật toán hoặc mô hình AI.</p>
<p>Nghiêm cấm:</p>
<ul>
<li>Chỉnh sửa thủ công từng ảnh bằng các phần mềm đồ họa</li>
<li>Ghép ảnh, vẽ thêm hoặc xóa vật thể bằng thao tác thủ công</li>
<li>Can thiệp thủ công vào từng test pose</li>
</ul>
<p>Ban Tổ Chức có quyền yêu cầu chứng minh quy trình sinh ảnh hoàn toàn tự động.</p>
<h1>11. Baseline thí sinh có thể tham khảo</h1>
<p><a href="https://github.com/graphdeco-inria/gaussian-splatting">https://github.com/graphdeco-inria/gaussian-splatting</a></p></div>
