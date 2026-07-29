::: {.html-header-logo}
[![logo](3dgs_deblur_paper_files/arxiv-logomark-small-white_JJYT.svg){.logomark
width="100"} [Back to arXiv]{.sr-only}](https://arxiv.org/)
:::

::: {.html-header-nav}
:::

::: {.html-header-logo}
[![logo](3dgs_deblur_paper_files/arxiv-logo-one-color-white_JJYT.svg){.logo
width="100"} [Back to arXiv]{.sr-only}](https://arxiv.org/)
:::

::: {.html-header-message role="banner"}
This is **experimental HTML** to improve accessibility. We invite you to
report rendering errors. [Use Alt+Y to toggle on accessible reporting
links and Alt+Shift+Y to toggle off.]{.sr-only} Learn more [about this
project](https://info.arxiv.org/about/accessible_HTML.html) and [help
improve
conversions](https://info.arxiv.org/help/submit_latex_best_practices.html).
:::

[Why
HTML?](https://info.arxiv.org/about/accessible_HTML.html){.ar5iv-footer-button
.hover-effect} [Report Issue](#myForm){.ar5iv-footer-button
.hover-effect} [Back to
Abstract](https://arxiv.org/abs/2403.13327v3){.ar5iv-footer-button
.hover-effect} [Download
PDF](https://arxiv.org/pdf/2403.13327v3){.ar5iv-footer-button
.hover-effect}

::: {#main .ltx_page_main}
Table of Contents {#toc_header .sr-only}
-----------------

::: {#listIcon .hide type="button"}
:::

::: {#arrowIcon type="button"}
:::

1.  [[ []{.ltx_tag .ltx_tag_ref} Abstract ]{.ltx_text
    .ltx_ref_title}](https://arxiv.org/html/2403.13327#abstract "Abstract"){.ltx_ref}
2.  [[[1 ]{.ltx_tag .ltx_tag_ref}Introduction]{.ltx_text
    .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S1 "In Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
3.  [[[2 ]{.ltx_tag .ltx_tag_ref}Related Work]{.ltx_text
    .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S2 "In Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
    1.  [[Image deblurring]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S2.SS0.SSS0.Px1 "In 2 Related Work ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
    2.  [[Deblurring 3D implicit representations]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S2.SS0.SSS0.Px2 "In 2 Related Work ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
4.  [[[3 ]{.ltx_tag .ltx_tag_ref}Methods]{.ltx_text
    .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S3 "In Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
    1.  [[[3.1 ]{.ltx_tag .ltx_tag_ref}Gaussian Splatting]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S3.SS1 "In 3 Methods ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
    2.  [[[3.2 ]{.ltx_tag .ltx_tag_ref}Blur and Rolling Shutter as
        Camera Motion]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S3.SS2 "In 3 Methods ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
    3.  [[[3.3 ]{.ltx_tag .ltx_tag_ref}Screen Space
        Approximation]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S3.SS3 "In 3 Methods ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
    4.  [[[3.4 ]{.ltx_tag .ltx_tag_ref}Rasterization with Pixel
        Velocities]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S3.SS4 "In 3 Methods ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
    5.  [[[3.5 ]{.ltx_tag .ltx_tag_ref}Pose Optimization]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S3.SS5 "In 3 Methods ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
    6.  [[[3.6 ]{.ltx_tag .ltx_tag_ref}Evaluation Strategy -
        Optimization with fixed Gaussians]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S3.SS6 "In 3 Methods ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
5.  [[[4 ]{.ltx_tag .ltx_tag_ref}Experiments]{.ltx_text
    .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S4 "In Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
    1.  [[[4.1 ]{.ltx_tag .ltx_tag_ref}Synthetic Data]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S4.SS1 "In 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
        1.  [[BAD-NeRF data set variant]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S4.SS1.SSS0.Px1 "In 4.1 Synthetic Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
        2.  [[Re-rendered data]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S4.SS1.SSS0.Px2 "In 4.1 Synthetic Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
    2.  [[[4.2 ]{.ltx_tag .ltx_tag_ref}Smartphone Data]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S4.SS2 "In 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
        1.  [[Preprocessing and VIO velocity estimation]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S4.SS2.SSS0.Px1 "In 4.2 Smartphone Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
        2.  [[Training and evaluation split]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S4.SS2.SSS0.Px2 "In 4.2 Smartphone Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
        3.  [[Pose and intrinsic estimation]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S4.SS2.SSS0.Px3 "In 4.2 Smartphone Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
        4.  [[COLMAP baseline]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S4.SS2.SSS0.Px4 "In 4.2 Smartphone Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
        5.  [[CVR de-rolling baseline]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S4.SS2.SSS0.Px5 "In 4.2 Smartphone Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
        6.  [[Ablation study]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S4.SS2.SSS0.Px6 "In 4.2 Smartphone Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
        7.  [[Results]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S4.SS2.SSS0.Px7 "In 4.2 Smartphone Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
        8.  [[Timing tests]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S4.SS2.SSS0.Px8 "In 4.2 Smartphone Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
6.  [[[5 ]{.ltx_tag .ltx_tag_ref}Discussion and Conclusion]{.ltx_text
    .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#S5 "In Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
7.  [[[0.A ]{.ltx_tag .ltx_tag_ref}Method Details]{.ltx_text
    .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#Pt0.A1 "In Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
    1.  [[[0.A.1 ]{.ltx_tag .ltx_tag_ref}Gaussian
        parametrization]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#Pt0.A1.SS1 "In Appendix 0.A Method Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
    2.  [[[0.A.2 ]{.ltx_tag .ltx_tag_ref}Transforming Gaussians from
        world to pixel coordinates]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#Pt0.A1.SS2 "In Appendix 0.A Method Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
    3.  [[[0.A.3 ]{.ltx_tag .ltx_tag_ref}Differentiation with respect to
        the camera pose]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#Pt0.A1.SS3 "In Appendix 0.A Method Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
    4.  [[[0.A.4 ]{.ltx_tag .ltx_tag_ref}Derivation of the pixel
        velocity formula]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#Pt0.A1.SS4 "In Appendix 0.A Method Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
    5.  [[[0.A.5 ]{.ltx_tag .ltx_tag_ref}Key Frame Selection]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#Pt0.A1.SS5 "In Appendix 0.A Method Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
    6.  [[[0.A.6 ]{.ltx_tag .ltx_tag_ref}Transferring velocities from
        one SLAM method to another]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#Pt0.A1.SS6 "In Appendix 0.A Method Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
8.  [[[0.B ]{.ltx_tag .ltx_tag_ref}Data Sets and Metrics]{.ltx_text
    .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#Pt0.A2 "In Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
    1.  [[Modifications to the Deblur-NeRF data set]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#Pt0.A2.SS0.SSS0.Px1 "In Appendix 0.B Data Sets and Metrics ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
9.  [[[0.C ]{.ltx_tag .ltx_tag_ref}Experiment Details]{.ltx_text
    .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#Pt0.A3 "In Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
    1.  [[Splatfacto hyperparameters]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#Pt0.A3.SS0.SSS0.Px1 "In Appendix 0.C Experiment Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
10. [[[0.D ]{.ltx_tag .ltx_tag_ref}Additional Results]{.ltx_text
    .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#Pt0.A4 "In Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
    1.  [[Figures]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#Pt0.A4.SS0.SSS0.Px1 "In Appendix 0.D Additional Results ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
    2.  [[[0.D.0.1 ]{.ltx_tag .ltx_tag_ref}Alternative
        intrinsics]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#Pt0.A4.SS0.SSS1 "In Appendix 0.D Additional Results ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
        1.  [[Blur-based key frame selection]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2403.13327v3#Pt0.A4.SS0.SSS1.Px1 "In 0.D.0.1 Alternative intrinsics ‣ Appendix 0.D Additional Results ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
11. [[ []{.ltx_tag .ltx_tag_ref} References ]{.ltx_text
    .ltx_ref_title}](https://arxiv.org/html/2403.13327#bib "References"){.ltx_ref}

::: {.ltx_page_content}
::: {.package-alerts .ltx_document role="status" aria-label="Conversion errors have been found" style="display: none;"}
HTML conversions [sometimes display
errors](https://info.dev.arxiv.org/about/accessibility_html_error_messages.html)
due to content that did not convert correctly from the source. This
paper uses the following packages that are not yet supported by the HTML
conversion tool. Feedback on these issues are not necessary; they are
known and are being worked on.

-   failed: changepage

Authors: achieve the best HTML results from your LaTeX submissions by
following these [best
practices](https://info.arxiv.org/help/submit_latex_best_practices.html).
:::

::: {#target-section .section}
[License: CC BY-SA
4.0](https://info.arxiv.org/help/license/index.html#licenses-available){#license-tr}

::: {#watermark-tr}
arXiv:2403.13327v3 \[cs.CV\] 17 Jul 2024
:::
:::

[^1^[[^1^[institutetext: ]{.ltx_note_type}^1^ Spectacular AI, ^2^ ETH
Zurich, ^3^ Aalto University\
^4^ University of Oulu, ^5^ Tampere University\
Corresponding author: [otto.seiskari\@spectacularai.com]{#id5.6
.ltx_text .ltx_font_typewriter
style="font-size:80%;"}]{.ltx_note_content}]{.ltx_note_outer}]{#id5
.ltx_note .ltx_role_institutetext}

Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion {#gaussian-splatting-on-the-move-blur-and-rolling-shutter-compensation-for-natural-camera-motion .ltx_title .ltx_title_document}
===============================================================================================

Report issue for preceding element

::: {.ltx_authors}
[ [Otto Seiskari^1^ ]{.ltx_personname}]{.ltx_creator .ltx_role_author}
[  ]{.ltx_author_before}[ [Jerry Ylilammi^1^
]{.ltx_personname}]{.ltx_creator .ltx_role_author}
[  ]{.ltx_author_before}[ [Valtteri Kaatrasalo^1^
]{.ltx_personname}]{.ltx_creator .ltx_role_author}
[  ]{.ltx_author_before}[ [Pekka Rantalankila^1^
]{.ltx_personname}]{.ltx_creator .ltx_role_author}
[  ]{.ltx_author_before}[ [Matias Turkulainen^[2,3]{#id18.2.id1.1
.ltx_text .ltx_font_italic}^ ]{.ltx_personname}]{.ltx_creator
.ltx_role_author} [  ]{.ltx_author_before}[ [Juho
Kannala^[1,3,4]{#id19.2.id1.1 .ltx_text .ltx_font_italic}^
]{.ltx_personname}]{.ltx_creator .ltx_role_author}
[  ]{.ltx_author_before}[ [Esa Rahtu^5^ ]{.ltx_personname}]{.ltx_creator
.ltx_role_author} [  ]{.ltx_author_before}[ [Arno
Solin^[1,3]{#id21.2.id1.1 .ltx_text .ltx_font_italic}^
]{.ltx_personname}]{.ltx_creator .ltx_role_author}
:::

Report issue for preceding element

::: {#abstract .ltx_abstract}
###### Abstract {#abstract .ltx_title .ltx_title_abstract}

Report issue for preceding element

High-quality scene reconstruction and novel view synthesis based on
Gaussian Splatting (3DGS) typically require steady, high-quality
photographs, often impractical to capture with handheld cameras. We
present a method that adapts to camera motion and allows high-quality
scene reconstruction with handheld video data suffering from motion blur
and rolling shutter distortion. Our approach is based on detailed
modelling of the physical image formation process and utilizes
velocities estimated using visual-inertial odometry (VIO). Camera poses
are considered non-static during the exposure time of a single image
frame and camera poses are further optimized in the reconstruction
process. We formulate a differentiable rendering pipeline that leverages
screen space approximation to efficiently incorporate rolling-shutter
and motion blur effects into the 3DGS framework. Our results with both
synthetic and real data demonstrate superior performance in mitigating
camera motion over existing methods, thereby advancing 3DGS in
naturalistic settings.

Report issue for preceding element
:::

::: {#p1 .ltx_para .ltx_noindent}
::: {#p1.1 .ltx_block .ltx_minipage .ltx_align_middle style="width:433.6pt;"}
![\[Uncaptioned
image\]](3dgs_deblur_paper_files/x1_JJYT.png){#p1.1.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.g1
.ltx_graphics .ltx_img_square width="245"
height="239"}[[p]{style="visibility:hidden"}]{#p1.1.pic1.6.6.6.1.1.1.1.1
.ltx_text .ltx_phantom
style="font-size:70%;"}[Input]{#p1.1.pic1.7.7.7.2.2.2.2.2 .ltx_text
style="font-size:70%;"}![\[Uncaptioned
image\]](3dgs_deblur_paper_files/x2_JJYT.png){#p1.1.pic1.2.2.2.2.2.2.2.2.2.2.2.2.1.1.1.1.1.1.1.1.1.g1
.ltx_graphics .ltx_img_square width="245"
height="239"}[[p]{style="visibility:hidden"}]{#p1.1.pic1.8.8.8.1.1.1.1.1
.ltx_text .ltx_phantom style="font-size:70%;"}[Motion
blur]{#p1.1.pic1.9.9.9.2.2.2.2.2 .ltx_text
style="font-size:70%;"}![\[Uncaptioned
image\]](3dgs_deblur_paper_files/x3_JJYT.png){#p1.1.pic1.3.3.3.3.3.3.3.3.3.3.3.3.1.1.1.1.1.1.1.1.1.g1
.ltx_graphics .ltx_img_square width="245"
height="239"}[[p]{style="visibility:hidden"}]{#p1.1.pic1.10.10.10.1.1.1.1.1
.ltx_text .ltx_phantom style="font-size:70%;"}[Rolling
shutter]{#p1.1.pic1.11.11.11.2.2.2.2.2 .ltx_text
style="font-size:70%;"}![\[Uncaptioned
image\]](3dgs_deblur_paper_files/x4_JJYT.png){#p1.1.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.g1
.ltx_graphics .ltx_img_square width="245"
height="239"}[[p]{style="visibility:hidden"}]{#p1.1.pic1.12.12.12.1.1.1.1.1
.ltx_text .ltx_phantom style="font-size:70%;"}[Our clean
render]{#p1.1.pic1.13.13.13.2.2.2.2.2 .ltx_text
style="font-size:70%;"}[Rolling shutter effects]{#p1.1.pic1.14.14.14.1.1
.ltx_text .ltx_font_italic style="font-size:50%;color:#808080;"}[Motion
blur]{#p1.1.pic1.15.15.15.1.1 .ltx_text
style="font-size:50%;color:#808080;"}[Motion
blur]{#p1.1.pic1.16.16.16.1.1 .ltx_text
style="font-size:50%;color:#BFBFBF;"}[Motion
blur]{#p1.1.pic1.17.17.17.1.1 .ltx_text
style="font-size:50%;color:#BFBFBF;"}![\[Uncaptioned
image\]](3dgs_deblur_paper_files/x5_JJYT.png){#p1.1.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.g1
.ltx_graphics .ltx_img_square width="301"
height="299"}[VIO]{#p1.1.pic1.18.18.18.1.1 .ltx_text
style="font-size:50%;"}

Report issue for preceding element
:::
:::

::: {#S1 .section .ltx_section}
[1 ]{.ltx_tag .ltx_tag_section}Introduction {#introduction .ltx_title .ltx_title_section}
-------------------------------------------

Report issue for preceding element

::: {#S1.p1 .ltx_para}
The field of novel view synthesis has seen significant advancements in
recent years, with the introduction of Neural Radiance Fields (NeRF,
\[[26](https://arxiv.org/html/2403.13327v3#bib.bib26){.ltx_ref}\]) and
more recently, Gaussian Splatting (3DGS,
\[[13](https://arxiv.org/html/2403.13327v3#bib.bib13){.ltx_ref}\]). Both
classes of methods represent scenes as differentiable, non-mesh-based,
3D representations that allow rendering of new views that are often
visually indistinguishable from evaluation images. One major limitation
of these methods is that they generally require high-quality still
photographs which can be accurately registered using photogrammetry
software, such as
COLMAP \[[31](https://arxiv.org/html/2403.13327v3#bib.bib31){.ltx_ref}\].\
Generating 3D reconstructions from casually captured data recorded with
a moving hand-held camera, such as a smartphone, would enable faster
data collection and integration into a broader range of use cases.
However, image data from a moving sensor is prone to motion blur and
rolling shutter distortion, which significantly degrade the quality of
the reconstruction and increases the likelihood of failures in pose
registration. Motion blur effects occur during the camera shutter's
opening time due to relative motion between the camera and objects in
the scene. Similarly, rolling shutter effects, caused by the camera
sensor scanning the scene line-by-line, lead to warping distortions in
fast-moving scenes or during rapid camera maneuvers.\
Most of the existing methods aiming to compensate for these imaging
effects use classical or deep learning--based approaches to recover
sharpened versions of the input images, without the aid of an underlying
3D image formation model for the scene. The recent 3D novel view
synthesis methods, including NeRF
 \[[26](https://arxiv.org/html/2403.13327v3#bib.bib26){.ltx_ref}\] and
3DGS  \[[13](https://arxiv.org/html/2403.13327v3#bib.bib13){.ltx_ref}\],
allow for an alternative approach where a sharp 3D reconstruction is
recovered without manipulating the training image data as an
intermediary step.
Deblur-NeRF \[[25](https://arxiv.org/html/2403.13327v3#bib.bib25){.ltx_ref}\],
and
BAD-NeRF \[[40](https://arxiv.org/html/2403.13327v3#bib.bib40){.ltx_ref}\]
perform this in the context of NeRFs. In the context of 3DGS,
\[[17](https://arxiv.org/html/2403.13327v3#bib.bib17){.ltx_ref}\]
propose a method for a related but different problem of *de-focus* blur
compensation. Similarly
to \[[40](https://arxiv.org/html/2403.13327v3#bib.bib40){.ltx_ref}\],
our work focuses on *camera motion blur* and applications to real data
where this is the most prominent blurring modality, such as hand-held
captures of static scenes without extreme close-up shots.\
This work offers an alternative deblurring and rolling shutter
correction approach for the 3DGS framework. Instead of learning blurring
kernels from the data, as in
\[[25](https://arxiv.org/html/2403.13327v3#bib.bib25){.ltx_ref}\], we
directly model the image formation process with camera motion and
rolling shutter effects leveraging velocity estimates computed using
visual-inertial odometry (VIO), a technique that fuses inertial
measurement unit (IMU) data with monocular video. We formulate an
efficient differentiable motion blur and rolling-shutter capable
rendering pipeline that utilizes screen space approximation to avoid
recomputing or hindering the performance of 3DGS operations. To address
the ill-posed nature of the deblurring problem (*cf*.[]{#S1.p1.1.4
.ltx_text} \[[11](https://arxiv.org/html/2403.13327v3#bib.bib11){.ltx_ref}\]),
we leverage the regularization capabilities of the differentiable 3DGS
framework priors information from sensor data.\
The performance of our method is evaluated using both synthetic data
from the Deblur-NeRF data set
\[[25](https://arxiv.org/html/2403.13327v3#bib.bib25){.ltx_ref}\] as
well as real-world data recorded using mobile devices. Our method is
implemented as an extension to the NerfStudio
\[[37](https://arxiv.org/html/2403.13327v3#bib.bib37){.ltx_ref}\] and
[gsplat]{#S1.p1.1.5 .ltx_text .ltx_font_typewriter}
\[[47](https://arxiv.org/html/2403.13327v3#bib.bib47){.ltx_ref}\]
software packages, which serve as our baseline methods for evaluation.
Our approach consistently outperforms the baselines for both synthetic
and real data experiments in terms of PSNR, SSIM, and LPIPS metrics, and
the resulting reconstructions appear qualitatively sharper.\

Report issue for preceding element
:::

![[[Figure 1]{#S1.F1.7.1.1 .ltx_text style="font-size:129%;"}:
]{.ltx_tag .ltx_tag_figure}[3DGS reconstructions from the synthetic
[cozyroom]{#S1.F1.8.2.1 .ltx_text .ltx_font_typewriter} scene under
different simulated effects with (bottom row) and without (top row) our
compensation. The corresponding numerical results are given in
[[Table]{.ltx_text .ltx_ref_tag} [1]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S4.T1 "In 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}.
The bottom right image is visually indistinguishable from the reference
(PSNR 36.2).]{#S1.F1.8.2 .ltx_text
style="font-size:129%;"}](3dgs_deblur_paper_files/synthetic-mb-baseline_JJYT.jpg){#S1.F1.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.g1
.ltx_graphics .ltx_img_landscape width="216" height="144"}

Report issue for preceding element

![[[Figure 2]{#S1.F2.7.1.1 .ltx_text style="font-size:129%;"}:
]{.ltx_tag .ltx_tag_figure}[Further 3DGS reconstructions from the
synthetic [factory]{#S1.F2.8.2.1 .ltx_text .ltx_font_typewriter} scene.
The Splatfacto method acting as a baseline. The corresponding numerical
results are given in [[Table]{.ltx_text .ltx_ref_tag} [1]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S4.T1 "In 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}.
In this case, the pose optimization case converged to a local
minimum.]{#S1.F2.8.2 .ltx_text
style="font-size:129%;"}](3dgs_deblur_paper_files/synthetic-factory-mb-baseline_JJYT.jpg){#S1.F2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.g1
.ltx_graphics .ltx_img_landscape width="216" height="144"}

Report issue for preceding element
:::

::: {#S2 .section .ltx_section}
[2 ]{.ltx_tag .ltx_tag_section}Related Work {#related-work .ltx_title .ltx_title_section}
-------------------------------------------

Report issue for preceding element

::: {#S2.p1 .ltx_para}
We cover classical and deep learning--based image deblurring and give an
overview of prior methods compensating for motion blur and rolling
shutter distortion in the context of 3D reconstruction using implicit
representations.

Report issue for preceding element
:::

::: {#S2.SS0.SSS0.Px1 .section .ltx_paragraph}
##### Image deblurring {#image-deblurring .ltx_title .ltx_title_paragraph}

Report issue for preceding element

::: {#S2.SS0.SSS0.Px1.p1 .ltx_para}
Motion blur compensation has been extensively studied in single and
multi-image settings. Classical approaches attempt to jointly recover
sharp input images along with blurring kernels with optimization based
methods
\[[30](https://arxiv.org/html/2403.13327v3#bib.bib30){.ltx_ref}\]. This
task is ill-posed since multiple blurring kernels can result in the same
blurred image, and prior work add regularization to account for this
\[[44](https://arxiv.org/html/2403.13327v3#bib.bib44){.ltx_ref},
[34](https://arxiv.org/html/2403.13327v3#bib.bib34){.ltx_ref},
[2](https://arxiv.org/html/2403.13327v3#bib.bib2){.ltx_ref}\].
Richardson--Lucy deblurring
\[[36](https://arxiv.org/html/2403.13327v3#bib.bib36){.ltx_ref}\]
attempts to account for spatially varying blurring with projective
homographies to describe three-dimensional camera motion. Deep
learning-based methods (*e.g*.[]{#S2.SS0.SSS0.Px1.p1.1.2 .ltx_text},
\[[3](https://arxiv.org/html/2403.13327v3#bib.bib3){.ltx_ref},
[15](https://arxiv.org/html/2403.13327v3#bib.bib15){.ltx_ref},
[8](https://arxiv.org/html/2403.13327v3#bib.bib8){.ltx_ref}\]) utilize
features learned on large training data sets to recover sharp images.
These methods outperform classical approaches that rely on hand crafted
image statistics and can better handle spatially varying deblurring.

Report issue for preceding element
:::

::: {#S2.SS0.SSS0.Px1.p2 .ltx_para}
Similarly, rolling shutter (RS) compensation has been traditionally
studied as a problem of optimally warping individual pixels or images to
account for the row-wise exposure of frames
\[[19](https://arxiv.org/html/2403.13327v3#bib.bib19){.ltx_ref},
[9](https://arxiv.org/html/2403.13327v3#bib.bib9){.ltx_ref},
[29](https://arxiv.org/html/2403.13327v3#bib.bib29){.ltx_ref},
[16](https://arxiv.org/html/2403.13327v3#bib.bib16){.ltx_ref}\]. Modern
methods such as
\[[39](https://arxiv.org/html/2403.13327v3#bib.bib39){.ltx_ref},
[22](https://arxiv.org/html/2403.13327v3#bib.bib22){.ltx_ref},
[6](https://arxiv.org/html/2403.13327v3#bib.bib6){.ltx_ref}\] utilize
information from three-dimensional image formation and camera motion to
better compensate for RS effects. Moreover, deep-learning based methods
trained on large data sets demonstrate occlusion and in-painting
capability, a feature that traditional methods lack. Rolling-shutter
compensation has also been studied in the scope of Structure-from-Motion
(SfM) and Visual-Inertial SLAM methods
\[[10](https://arxiv.org/html/2403.13327v3#bib.bib10){.ltx_ref},
[23](https://arxiv.org/html/2403.13327v3#bib.bib23){.ltx_ref},
[32](https://arxiv.org/html/2403.13327v3#bib.bib32){.ltx_ref},
[20](https://arxiv.org/html/2403.13327v3#bib.bib20){.ltx_ref}\].

Report issue for preceding element
:::

::: {#S2.SS0.SSS0.Px1.p3 .ltx_para}
Simultaneous motion blur and rolling shutter compensation has also been
studied, in, *e.g*.[]{#S2.SS0.SSS0.Px1.p3.1.2 .ltx_text},
\[[35](https://arxiv.org/html/2403.13327v3#bib.bib35){.ltx_ref},
[27](https://arxiv.org/html/2403.13327v3#bib.bib27){.ltx_ref}\].
Learning-based methods aimed at generic image blur and artefacts, such
as \[[50](https://arxiv.org/html/2403.13327v3#bib.bib50){.ltx_ref}\],
have also improved significantly and were used in
\[[25](https://arxiv.org/html/2403.13327v3#bib.bib25){.ltx_ref}\] as a
pre-processing step to enhance NeRF reconstruction.

Report issue for preceding element
:::
:::

::: {#S2.SS0.SSS0.Px2 .section .ltx_paragraph}
##### Deblurring 3D implicit representations {#deblurring-3d-implicit-representations .ltx_title .ltx_title_paragraph}

Report issue for preceding element

::: {#S2.SS0.SSS0.Px2.p1 .ltx_para}
Deblurring differentiable implicit representations for novel view
synthesis is a relatively new topic and has been previously studied in
\[[25](https://arxiv.org/html/2403.13327v3#bib.bib25){.ltx_ref},
[5](https://arxiv.org/html/2403.13327v3#bib.bib5){.ltx_ref},
[40](https://arxiv.org/html/2403.13327v3#bib.bib40){.ltx_ref}\] for NeRF
\[[26](https://arxiv.org/html/2403.13327v3#bib.bib26){.ltx_ref}\]
representations and
\[[17](https://arxiv.org/html/2403.13327v3#bib.bib17){.ltx_ref}\] in the
3DGS \[[13](https://arxiv.org/html/2403.13327v3#bib.bib13){.ltx_ref}\]
context. Both
\[[25](https://arxiv.org/html/2403.13327v3#bib.bib25){.ltx_ref}\] and
\[[17](https://arxiv.org/html/2403.13327v3#bib.bib17){.ltx_ref}\]
incorporate additional learnable parameters that model blurring effects
as part of the rendering pipeline in the form of small Multi-Layer
Perceptrons (MLPs) added to the baseline models. These methods treat the
training images as fixed in time and focus on extracting sharper images
from blurred inputs. An alternative method is to model the per frame
image formation model by accounting for the motion trajectory during the
capture process as presented in
\[[40](https://arxiv.org/html/2403.13327v3#bib.bib40){.ltx_ref}\]. Our
approach is most similar to
\[[40](https://arxiv.org/html/2403.13327v3#bib.bib40){.ltx_ref}\] in the
sense that we explicitly model the blur formation process by integrating
information over a short camera trajectory. Rolling shutter effects,
which are not considered in the aforementioned works, is separately
considered in the NeRF context in
\[[18](https://arxiv.org/html/2403.13327v3#bib.bib18){.ltx_ref}\], which
also incorporates additional learnable parameters to the training
process. In contrast, our approach does not include additional MLPs to
the 3DGS pipeline, but directly models the local camera trajectory using
linear and angular velocities, for which good initial estimates are
readily available from IMU data captured from Visual-Inertial Odometry
(VIO) pipelines.

Report issue for preceding element
:::

::: {#S2.SS0.SSS0.Px2.p2 .ltx_para}
Our method also utilizes pose optimization in the 3DGS framework,
primarily for better pose registration in the presence of rolling
shutter effects, which are not efficiently handled by
COLMAP \[[31](https://arxiv.org/html/2403.13327v3#bib.bib31){.ltx_ref}\].
Pose refinement can also contribute to sharpening the reconstruction
quality by effectively mitigating deblurring caused by the sensitivity
of NeRF and 3DGS based methods on accurate pose estimates. Pose
refinement has been previously applied to 3DGS in
\[[45](https://arxiv.org/html/2403.13327v3#bib.bib45){.ltx_ref},
[12](https://arxiv.org/html/2403.13327v3#bib.bib12){.ltx_ref},
[matsuki2023gaussian]{.ltx_ref .ltx_missing_citation .ltx_ref_self},
[7](https://arxiv.org/html/2403.13327v3#bib.bib7){.ltx_ref}\], but not
together with rolling shutter compensation. Correspondingly, pose
optimization with NeRFs has been studied in several works
(*e.g*.[]{#S2.SS0.SSS0.Px2.p2.1.2
.ltx_text}, \[[28](https://arxiv.org/html/2403.13327v3#bib.bib28){.ltx_ref},
[40](https://arxiv.org/html/2403.13327v3#bib.bib40){.ltx_ref}\]),
starting with
BARF \[[21](https://arxiv.org/html/2403.13327v3#bib.bib21){.ltx_ref}\].
Utilizing information from VIO in NeRFs to stabilize pose optimization
has been studied in
\[[14](https://arxiv.org/html/2403.13327v3#bib.bib14){.ltx_ref}\], but
to the best of our knowledge, has not been studied in the 3DGS context.

Report issue for preceding element
:::

::: {#S2.SS0.SSS0.Px2.p3 .ltx_para}
Our implementation is based on Nerfstudio
\[[37](https://arxiv.org/html/2403.13327v3#bib.bib37){.ltx_ref}\] and
[gsplat]{#S2.SS0.SSS0.Px2.p3.1.1 .ltx_text .ltx_font_typewriter}
\[[47](https://arxiv.org/html/2403.13327v3#bib.bib47){.ltx_ref}\]. For
robust VIO, we utilize the Spectacular AI SDK, a proprietary VISLAM
system loosely based
on \[[33](https://arxiv.org/html/2403.13327v3#bib.bib33){.ltx_ref}\].

Report issue for preceding element
:::
:::
:::

::: {#S3 .section .ltx_section}
[3 ]{.ltx_tag .ltx_tag_section}Methods {#methods .ltx_title .ltx_title_section}
--------------------------------------

Report issue for preceding element

::: {#S3.p1 .ltx_para}
We start with a brief overview of Gaussian Splatting and establishing
notation, followed by our formulation of motion blur and rolling shutter
effects in the 3DGS framework. For this, we also introduce a novel
screen space approximation and pixel velocity-based rasterization,
concluding with pose optimization and its role in the evaluation
methodology.

Report issue for preceding element
:::

::: {#S3.SS1 .section .ltx_subsection}
### [3.1 ]{.ltx_tag .ltx_tag_subsection}Gaussian Splatting {#gaussian-splatting .ltx_title .ltx_title_subsection}

Report issue for preceding element

::: {#S3.SS1.p1 .ltx_para}
Gaussian Splatting (3DGS,
\[[13](https://arxiv.org/html/2403.13327v3#bib.bib13){.ltx_ref}\])
serves as an example and current cornerstone of *differentiable
rendering*, enabling the differentiation of a pixel's colour in relation
to model parameters. This capability allows for the optimization of a
loss function by comparing rendered images to real reference images. In
essence, Gaussian Splatting maps the 3D scene into 2D images through a
set of Gaussian distributions with additional colour and transparency
attributes, each contributing to the region of the scene defined by the
its mean and covariance.

Report issue for preceding element
:::

::: {#S3.SS1.p2 .ltx_para}
In Gaussian Splatting, the colour

Report issue for preceding element

  -- ------------------------------------------------------ -- ----------------------------------------------------
     $${C_{i}{(x,y,P_{i},\mathcal{G})}} \in \mathcal{C}$$      [(1)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- ------------------------------------------------------ -- ----------------------------------------------------

of the pixel ${(x,y)} \in {{\lbrack 0,W)} \times {\lbrack 0,H)}}$ of
output image $i$ can be differentiated with respect to the model
parameters $\mathcal{G}$, and these parameters can be used to optimize a
loss function

Report issue for preceding element

  -- ---------------------------------------------------------------------------------------------------------------------------------------------------------- -- ----------------------------------------------------
     $$\mathcal{G}\mapsto{\sum\limits_{i = 1}^{N_{img}}{\mathcal{L}\left\lbrack {C_{i}{( \cdot , \cdot ,P_{i},\mathcal{G})}},C_{i}^{\prime} \right\rbrack}}$$      [(2)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- ---------------------------------------------------------------------------------------------------------------------------------------------------------- -- ----------------------------------------------------

comparing the rendered images $C_{i}$ to a set of $N_{img}$ real
reference images $C_{i}^{\prime}$. We denote by $P_{i} \in {{SE}{(3)}}$
camera pose corresponding to image $i$ and assume
$\mathcal{C} = {\mathbb{R}}^{3}$ for RGB colours.

Report issue for preceding element
:::

::: {#S3.SS1.p3 .ltx_para}
The 3DGS model parameters consist of a set

Report issue for preceding element

  -- -------------------------------------------------------------------------- --
     $$\mathcal{G} = {\{{(\mu_{j},\Sigma_{j},\alpha_{j},\theta_{j})}\}}_{j}$$   
  -- -------------------------------------------------------------------------- --

of Gaussian distributions $(\mu_{j},\Sigma_{j})$, transparencies
$\alpha_{j}$ and view-dependent colours
$\theta_{j} \in \mathcal{C}^{N_{sh}}$ represented by $N_{sh}$ spherical
harmonic coefficient vectors.

Report issue for preceding element
:::
:::

::: {#S3.SS2 .section .ltx_subsection}
### [3.2 ]{.ltx_tag .ltx_tag_subsection}Blur and Rolling Shutter as Camera Motion {#blur-and-rolling-shutter-as-camera-motion .ltx_title .ltx_title_subsection}

Report issue for preceding element

::: {#S3.SS2.p1 .ltx_para}
We simultaneously model motion blur and rolling shutter as dynamic
three-dimensional effects caused by the motion of the camera along a
continuous trajectory $t\mapsto{P{(t)}}$. In terms of 3DGS, this can be
modelled by changing the rendering equation to

Report issue for preceding element

  -- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- ----------------------------------------------------
     $${{C_{i}{(x,y,\mathcal{G})}} = {g\left( {\frac{1}{T_{e}}{\int_{- {\frac{1}{2}T_{e}}}^{\frac{1}{2}T_{e}}{C_{i}\left( x,y,{P\left( {t_{i} + t_{e} + {{({{y/H} - {1/2}})}T_{ro}}} \right)},\mathcal{G} \right){dt_{e}}}}} \right)}},$$      [(3)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- ----------------------------------------------------

where $T_{ro}$ is the rolling-shutter readout time, $T_{e}$ the exposure
time, and $t_{i}$ the frame midpoint timestamp.
Following \[[25](https://arxiv.org/html/2403.13327v3#bib.bib25){.ltx_ref}\],
we assume a simple gamma correction model:
${g{(R,G,B)}} = {(R^{1/\gamma},G^{1/\gamma},B^{1/\gamma})}$ with
$\gamma = 2.2$. As a result, the colors $\theta_{j}$ of the splats are
defined in *linear RGB* space, whereas the original version
([[Eq.]{.ltx_text .ltx_ref_tag} [1]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S3.E1 "In 3.1 Gaussian Splatting ‣ 3 Methods ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref})
models them directly in the gamma-corrected colour space.

Report issue for preceding element
:::

::: {#S3.SS2.p2 .ltx_para}
We model the camera motion around the frame midpoint time $t_{i}$ as

Report issue for preceding element

  -- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- ----------------------------------------------------
     $${{P{({t_{i} + {\Delta t}})}} = {{\lbrack\left. R \middle| p \right.\rbrack}{(t)}} = \left\lbrack {R_{i}{\exp{({\Delta t{\lbrack\omega_{i}\rbrack}_{\times}})}}} \middle| {p_{i} + {{{\Delta t} \cdot R_{i}}v_{i}}} \right\rbrack},$$      [(4)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- ----------------------------------------------------

where $(v_{i},\omega_{i})$ are the linear and angular velocities of the
camera in its local coordinate system (assumed constant throughout the
frame interval ${|{\Delta t}|} < {\frac{1}{2}{({T_{e} + T_{ro}})}}$). We
add the velocities as additional optimizable parameters and set their
initial values to the estimate from VIO, if available, or zero
otherwise.

Report issue for preceding element
:::

![[[Figure 3]{#S3.F3.2.1.1 .ltx_text style="font-size:90%;"}: ]{.ltx_tag
.ltx_tag_figure}[Screen space approximation incorporates the motion
during the frame exposure interval into our model by capturing its
effect in pixel coordinates in the image plane. In our approach, the
rendering model of 3DGS is decomposed into two stages: first,
transforming the Gaussian parameters from world to pixel coordinates,
and then rasterizing these parameters onto the image.]{#S3.F3.3.2
.ltx_text
style="font-size:90%;"}](3dgs_deblur_paper_files/frustum_JJYT.png){#S3.F3.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.g1
.ltx_graphics .ltx_img_square width="36" height="36"}

Report issue for preceding element
:::

::: {#S3.SS3 .section .ltx_subsection}
### [3.3 ]{.ltx_tag .ltx_tag_subsection}Screen Space Approximation {#screen-space-approximation .ltx_title .ltx_title_subsection}

Report issue for preceding element

::: {#S3.SS3.p1 .ltx_para}
In our approach, the rendering model of 3DGS is decomposed into two
stages: first, transforming the Gaussian parameters from world to pixel
coordinates, and then rasterizing these parameters onto the image. This
transformation is pivotal for incorporating motion effects due to camera
movement during frame capture. By approximating the motion in pixel
coordinates, we focus on adjusting the Gaussian means to reflect this
movement, simplifying the model by primarily altering these means while
keeping other parameters stable. A high-level overview of the process is
presented in [[Fig.]{.ltx_text .ltx_ref_tag} [3]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S3.F3 "In 3.2 Blur and Rolling Shutter as Camera Motion ‣ 3 Methods ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}.

Report issue for preceding element
:::

::: {#S3.SS3.p2 .ltx_para}
These two main phases in the 3DGS rendering model can be writen as
${C_{i}{(x,y,\mathcal{G},P_{i})}} = {r{(x,y,{p{(\mathcal{G},P_{i})}})}}$,
where $p$ maps the Gaussian parameters $\mathcal{G}$ defined in world
coordinates to parameters
$\{{\mathcal{G}_{i,j}^{\prime} = {(\mu_{i,j}^{\prime},d_{i,j},\Sigma_{i,j}^{\prime},\alpha_{j},c_{i,j})}}\}$
defined in pixel coordinates of the currently processed camera $i$ and
individual Gaussian $j$ (see [[Sec.]{.ltx_text
.ltx_ref_tag} [0.A.2]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A1.SS2 "0.A.2 Transforming Gaussians from world to pixel coordinates ‣ Appendix 0.A Method Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
for details). The subsequent rasterization phase $r$ does not depend on
the camera pose $P_{i}$.

Report issue for preceding element
:::

::: {#S3.SS3.p3 .ltx_para}
We incorporate camera motion during the frame exposure interval into our
model by approximating its effect in pixel coordinates as

Report issue for preceding element

  -- ----------------------------------------------------------------------------------------------------------------------------------------------------------------- -- ----------------------------------------------------
     $${\mu_{i,j}^{\prime}{({\Delta t})}} \approx {{\overset{\sim}{\mu}}_{i,j}{({\Delta t})}}:={{\mu_{i,j}^{\prime}{(0,0)}} + {{\Delta t} \cdot v_{i,j}^{\prime}}}$$      [(5)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- ----------------------------------------------------------------------------------------------------------------------------------------------------------------- -- ----------------------------------------------------

in particular, to simplify and optimize the implementation, we neglect
the effect of (small) camera motion on the other view-dependent
intermediary variables $(d_{i,j},\Sigma_{i,j}^{\prime},c_{i,j})$ and
only model the effect on the pixel coordinates $\mu_{i,j}^{\prime}$ of
the Gaussian means and introduce a new set of variables, the *pixel
velocities* computed as

Report issue for preceding element

  -- ------------------------------------------------------------------------------------------ -- ----------------------------------------------------
     $${v_{i,j}^{\prime} = {- {J_{i}{({{\omega_{i} \times {\hat{\mu}}_{i,j}} + v_{i}})}}}},$$      [(6)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- ------------------------------------------------------------------------------------------ -- ----------------------------------------------------

where $J_{i} = {{{diag}{(f_{x},f_{y})}}/d_{i,j}}$ is the Jacobian of the
projective transform as in
\[[13](https://arxiv.org/html/2403.13327v3#bib.bib13){.ltx_ref}\] (see
[[Sec.]{.ltx_text .ltx_ref_tag} [0.A.4]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A1.SS4 "0.A.4 Derivation of the pixel velocity formula ‣ Appendix 0.A Method Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
for details).

Report issue for preceding element
:::
:::

::: {#S3.SS4 .section .ltx_subsection}
### [3.4 ]{.ltx_tag .ltx_tag_subsection}Rasterization with Pixel Velocities {#rasterization-with-pixel-velocities .ltx_title .ltx_title_subsection}

Report issue for preceding element

::: {#S3.SS4.p1 .ltx_para}
We approximate the integral in [[Eq.]{.ltx_text
.ltx_ref_tag} [3]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S3.E3 "In 3.2 Blur and Rolling Shutter as Camera Motion ‣ 3 Methods ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
with a sum of $N_{blur}$ samples on a fixed uniform grid during the
exposure interval

Report issue for preceding element

  -- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- ----------------------------------------------------
     $${{{\overset{\sim}{C}}_{i}{(x,y,\mathcal{G})}}:={g\left( {\frac{1}{N_{blur}}{\sum\limits_{k = 1}^{N_{blur}}{{\overset{\sim}{C}}_{i}\left( x,y,\mathcal{G},{\Delta t_{k}{(y)}} \right)}}} \right)}},$$      [(7)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- ----------------------------------------------------

where

Report issue for preceding element

  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- ----------------------------------------------------
     $${{\Delta t_{k}{(y)}}:={{\left( {\frac{k - 1}{N_{blur} - 1} - \frac{1}{2}} \right) \cdot T_{e}} + {\left( {\frac{y}{H} - \frac{1}{2}} \right) \cdot T_{ro}}}}.$$      [(8)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- ----------------------------------------------------

The 3DGS alpha blending stage (*cf*.[]{#S3.SS4.p1.5.2
.ltx_text} \[[13](https://arxiv.org/html/2403.13327v3#bib.bib13){.ltx_ref}\])
can be written as the sum
${C{(x,y,\mathcal{G})}} = {\sum_{j}{T_{j}c_{i,j}\alpha_{i,j}{(\mu_{i,j}^{\prime})}}}$
over the depth-sorted Gaussians, where
$T_{j} = {\prod_{l = 1}^{j - 1}{({1 - {\alpha_{i,l}{(\mu_{i,l}^{\prime})}}})}}$
is the accumulated transmittance. The dependency on other Gaussian pixel
space parameters $\mathcal{G}_{i,j}^{\prime}$ other than
$\mu_{i,j}^{\prime}$ has been omitted as explained in  [[Sec.]{.ltx_text
.ltx_ref_tag} [3.3]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S3.SS3 "3.3 Screen Space Approximation ‣ 3 Methods ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}.
With this, [[Eq.]{.ltx_text .ltx_ref_tag} [7]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S3.E7 "In 3.4 Rasterization with Pixel Velocities ‣ 3 Methods ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
becomes

Report issue for preceding element

  -- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- ----------------------------------------------------
     $${{{\overset{\sim}{C}}_{i}{(x,y,\mathcal{G})}}:={g\left( {\frac{1}{N_{blur}}{\sum\limits_{k = 1}^{N_{blur}}{\sum\limits_{j}{T_{j}c_{i,j}\alpha_{i,j}{({{\overset{\sim}{\mu}}_{i,j}^{\prime}{({\Delta t_{k}{(y)}})}})}}}}} \right)}}.$$      [(9)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- ----------------------------------------------------

Approximating camera velocity on the Gaussian means in pixel-space has
the benefit that the projective transform of the 3DGS pipeline does not
need to be repeated to generate each blur sample
(*i.e*.[]{#S3.SS4.p1.7.2 .ltx_text} the terms in the outer sum), making
the method faster than treating Gaussian velocities in world space.

Report issue for preceding element
:::
:::

::: {#S3.SS5 .section .ltx_subsection}
### [3.5 ]{.ltx_tag .ltx_tag_subsection}Pose Optimization {#pose-optimization .ltx_title .ltx_title_subsection}

Report issue for preceding element

::: {#S3.SS5.p1 .ltx_para}
We approximate the gradient with respect to camera pose components as

Report issue for preceding element

  -- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----------------------------------------------------
     $${{\frac{\partial C_{i}}{\partial p_{i}} \approx {{- {\sum\limits_{j}\frac{\partial C_{i}}{\partial\mu_{j}}}}\quad\text{and}}}\quad{\frac{\partial C_{i}}{\partial\nu} \approx {- {\sum\limits_{j}{\frac{\partial C_{i}}{\partial\mu_{j}}\frac{\partial R_{i}}{\partial\nu}R_{i}^{\top}{({\mu_{j} - p_{i}})}}}}}},$$      [(10)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----------------------------------------------------

where $\nu$ is any component of $R_{i}$ or a parameter it depends on.
This neglects the effect of small camera motions on view-dependent
colours and Gaussian precisions $\Sigma_{i,j}^{\prime}$ defined in pixel
coordinates, but allows approximating the expression in terms of the
Gaussian position derivatives
$\frac{\partial C_{i}}{\partial\mu_{j,k}}$, which are readily available
in the 3DGS pipeline. This approach is similar to
\[[45](https://arxiv.org/html/2403.13327v3#bib.bib45){.ltx_ref}\] and
derived in more detail in [[Sec.]{.ltx_text
.ltx_ref_tag} [0.A.3]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A1.SS3 "0.A.3 Differentiation with respect to the camera pose ‣ Appendix 0.A Method Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}.
To stabilize the reconstruction, we employ simple $l_{2}$ penalty terms
to prevent the poses from drifting too far from their initial estimates.

Report issue for preceding element
:::
:::

::: {#S3.SS6 .section .ltx_subsection}
### [3.6 ]{.ltx_tag .ltx_tag_subsection}Evaluation Strategy - Optimization with fixed Gaussians {#evaluation-strategy---optimization-with-fixed-gaussians .ltx_title .ltx_title_subsection}

Report issue for preceding element

::: {#S3.SS6.p1 .ltx_para}
Due to the well-known gauge invariance studied in classical
photogrammetry (*cf*.[]{#S3.SS6.p1.4.2
.ltx_text} \[[38](https://arxiv.org/html/2403.13327v3#bib.bib38){.ltx_ref}\]),
the inverse rendering problem has seven undetermined degrees of freedom:
rotating, scaling and translating the coordinate system in all the
quantities $(\mathcal{G},{\{ P_{i},v_{i},\omega_{i}\}}_{i})$ in
consistent manner yields new solution
$(\mathcal{G}^{\prime},{\{ P_{i}^{\prime},v_{i}^{\prime},\omega_{i}^{\prime}\}}_{i})$
with equal photometric losses. While, the $l_{2}$ penalty on the poses
will fix this gauge invariance, it leaves room for small perturbations:
the reconstructed scene may slightly rotate and translate, which can
cause the *evaluation frames* to be misaligned with reconstruction where
the training frames and Gaussians $\mathcal{G}$ are consistent with each
other.

Report issue for preceding element
:::

::: {#S3.SS6.p2 .ltx_para}
To tackle this, we also optimize the poses and velocities of the
evaluation frames with fixed gaussians $\mathcal{G}$, which can be
achieved by including both test and training frames to the optimization
problem, but disabling the back-propagation of gradient data to the
parameters $\mathcal{G}$ (while allowing it for $P_{i},v_{i}$ and
$\omega_{i}$) whenever an evaluation index $i$ is selected by the SGD
optimizer.

Report issue for preceding element
:::
:::
:::

::: {#S4 .section .ltx_section}
[4 ]{.ltx_tag .ltx_tag_section}Experiments {#experiments .ltx_title .ltx_title_section}
------------------------------------------

Report issue for preceding element

::: {#S4.p1 .ltx_para}
We evaluate the performance of our method on two different data sets:
synthetic data based on the Deblur-NeRF data
set \[[25](https://arxiv.org/html/2403.13327v3#bib.bib25){.ltx_ref}\]
and real data recorded using mobile phones. We implemented our method as
an extension to the open source 3DGS implementation *Splatfacto* in
Nerfstudio \[[37](https://arxiv.org/html/2403.13327v3#bib.bib37){.ltx_ref}\],
which is based on [gsplat]{#S4.p1.1.2 .ltx_text
.ltx_font_typewriter} \[[47](https://arxiv.org/html/2403.13327v3#bib.bib47){.ltx_ref}\]
(*cf*.[]{#S4.p1.1.4 .ltx_text} [[App.]{.ltx_text
.ltx_ref_tag} [0.C]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A3 "Appendix 0.C Experiment Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
for details).

Report issue for preceding element
:::

[[Table 1]{#S4.T1.16.1.1 .ltx_text style="font-size:129%;"}: ]{.ltx_tag
.ltx_tag_table}[Synthetic data results comparing the baseline Splatfacto
method to our approach with the appropriate compensation for each
variation.]{#S4.T1.17.2 .ltx_text style="font-size:129%;"}
:::
:::
:::

[Cozyroom]{#S4.T1.12.13.1.2.1 .ltx_text style="font-size:70%;"}

[Factory]{#S4.T1.12.13.1.3.1 .ltx_text style="font-size:70%;"}

[Pool]{#S4.T1.12.13.1.4.1 .ltx_text style="font-size:70%;"}

[Tanabata]{#S4.T1.12.13.1.5.1 .ltx_text style="font-size:70%;"}

[PSNR$\uparrow$]{#S4.T1.1.1.1.1 .ltx_text style="font-size:50%;"}

[SSIM$\uparrow$]{#S4.T1.2.2.2.1 .ltx_text style="font-size:50%;"}

[LPIPS$\downarrow$]{#S4.T1.3.3.3.1 .ltx_text style="font-size:50%;"}

[PSNR$\uparrow$]{#S4.T1.4.4.4.1 .ltx_text style="font-size:50%;"}

[SSIM$\uparrow$]{#S4.T1.5.5.5.1 .ltx_text style="font-size:50%;"}

[LPIPS$\downarrow$]{#S4.T1.6.6.6.1 .ltx_text style="font-size:50%;"}

[PSNR$\uparrow$]{#S4.T1.7.7.7.1 .ltx_text style="font-size:50%;"}

[SSIM$\uparrow$]{#S4.T1.8.8.8.1 .ltx_text style="font-size:50%;"}

[LPIPS$\downarrow$]{#S4.T1.9.9.9.1 .ltx_text style="font-size:50%;"}

[PSNR$\uparrow$]{#S4.T1.10.10.10.1 .ltx_text style="font-size:50%;"}

[SSIM$\uparrow$]{#S4.T1.11.11.11.1 .ltx_text style="font-size:50%;"}

[LPIPS$\downarrow$]{#S4.T1.12.12.12.1 .ltx_text style="font-size:50%;"}

[]{.ltx_rule
style="width:0.0pt;height:7.0pt;background:black;display:inline-block;"}[Motion
blur]{#S4.T1.12.14.2.1.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[Baseline]{#S4.T1.12.15.3.1.1 .ltx_text style="font-size:70%;"}

[26.63]{#S4.T1.12.15.3.2.1 .ltx_text style="font-size:70%;"}

[.832]{#S4.T1.12.15.3.3.1 .ltx_text style="font-size:70%;"}

[.190]{#S4.T1.12.15.3.4.1 .ltx_text style="font-size:70%;"}

[22.26]{#S4.T1.12.15.3.5.1 .ltx_text style="font-size:70%;"}

[.643]{#S4.T1.12.15.3.6.1 .ltx_text style="font-size:70%;"}

[.357]{#S4.T1.12.15.3.7.1 .ltx_text style="font-size:70%;"}

[35.50]{#S4.T1.12.15.3.8.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.951]{#S4.T1.12.15.3.9.1 .ltx_text style="font-size:70%;"}

[.046]{#S4.T1.12.15.3.10.1 .ltx_text style="font-size:70%;"}

[20.43]{#S4.T1.12.15.3.11.1 .ltx_text style="font-size:70%;"}

[.698]{#S4.T1.12.15.3.12.1 .ltx_text style="font-size:70%;"}

[.319]{#S4.T1.12.15.3.13.1 .ltx_text style="font-size:70%;"}

[Ours]{#S4.T1.12.16.4.1.1 .ltx_text style="font-size:70%;"}

[32.20]{#S4.T1.12.16.4.2.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.942]{#S4.T1.12.16.4.3.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.030]{#S4.T1.12.16.4.4.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[30.67]{#S4.T1.12.16.4.5.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.936]{#S4.T1.12.16.4.6.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.044]{#S4.T1.12.16.4.7.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[35.27]{#S4.T1.12.16.4.8.1 .ltx_text style="font-size:70%;"}

[.953]{#S4.T1.12.16.4.9.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.039]{#S4.T1.12.16.4.10.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[26.71]{#S4.T1.12.16.4.11.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.909]{#S4.T1.12.16.4.12.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.068]{#S4.T1.12.16.4.13.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[]{.ltx_rule
style="width:0.0pt;height:7.0pt;background:black;display:inline-block;"}[Rolling
shutter effect]{#S4.T1.12.17.5.1.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[Baseline]{#S4.T1.12.18.6.1.1 .ltx_text style="font-size:70%;"}

[19.21]{#S4.T1.12.18.6.2.1 .ltx_text style="font-size:70%;"}

[.627]{#S4.T1.12.18.6.3.1 .ltx_text style="font-size:70%;"}

[.184]{#S4.T1.12.18.6.4.1 .ltx_text style="font-size:70%;"}

[15.29]{#S4.T1.12.18.6.5.1 .ltx_text style="font-size:70%;"}

[.335]{#S4.T1.12.18.6.6.1 .ltx_text style="font-size:70%;"}

[.338]{#S4.T1.12.18.6.7.1 .ltx_text style="font-size:70%;"}

[27.32]{#S4.T1.12.18.6.8.1 .ltx_text style="font-size:70%;"}

[.776]{#S4.T1.12.18.6.9.1 .ltx_text style="font-size:70%;"}

[.083]{#S4.T1.12.18.6.10.1 .ltx_text style="font-size:70%;"}

[13.42]{#S4.T1.12.18.6.11.1 .ltx_text style="font-size:70%;"}

[.349]{#S4.T1.12.18.6.12.1 .ltx_text style="font-size:70%;"}

[.410]{#S4.T1.12.18.6.13.1 .ltx_text style="font-size:70%;"}

[Ours]{#S4.T1.12.19.7.1.1 .ltx_text style="font-size:70%;"}

[35.84]{#S4.T1.12.19.7.2.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.979]{#S4.T1.12.19.7.3.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.013]{#S4.T1.12.19.7.4.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[35.27]{#S4.T1.12.19.7.5.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.983]{#S4.T1.12.19.7.6.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.009]{#S4.T1.12.19.7.7.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[35.11]{#S4.T1.12.19.7.8.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.957]{#S4.T1.12.19.7.9.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.036]{#S4.T1.12.19.7.10.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[26.03]{#S4.T1.12.19.7.11.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.918]{#S4.T1.12.19.7.12.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.046]{#S4.T1.12.19.7.13.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[]{.ltx_rule
style="width:0.0pt;height:7.0pt;background:black;display:inline-block;"}[Pose
noise]{#S4.T1.12.20.8.1.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[Baseline]{#S4.T1.12.21.9.1.1 .ltx_text style="font-size:70%;"}

[16.76]{#S4.T1.12.21.9.2.1 .ltx_text style="font-size:70%;"}

[.484]{#S4.T1.12.21.9.3.1 .ltx_text style="font-size:70%;"}

[.339]{#S4.T1.12.21.9.4.1 .ltx_text style="font-size:70%;"}

[15.09]{#S4.T1.12.21.9.5.1 .ltx_text style="font-size:70%;"}

[.251]{#S4.T1.12.21.9.6.1 .ltx_text style="font-size:70%;"}

[.420]{#S4.T1.12.21.9.7.1 .ltx_text style="font-size:70%;"}

[20.95]{#S4.T1.12.21.9.8.1 .ltx_text style="font-size:70%;"}

[.465]{#S4.T1.12.21.9.9.1 .ltx_text style="font-size:70%;"}

[.344]{#S4.T1.12.21.9.10.1 .ltx_text style="font-size:70%;"}

[13.97]{#S4.T1.12.21.9.11.1 .ltx_text style="font-size:70%;"}

[.271]{#S4.T1.12.21.9.12.1 .ltx_text style="font-size:70%;"}

[.442]{#S4.T1.12.21.9.13.1 .ltx_text style="font-size:70%;"}

[Ours]{#S4.T1.12.22.10.1.1 .ltx_text style="font-size:70%;"}

[36.30]{#S4.T1.12.22.10.2.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.976]{#S4.T1.12.22.10.3.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.013]{#S4.T1.12.22.10.4.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[24.03]{#S4.T1.12.22.10.5.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.745]{#S4.T1.12.22.10.6.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.132]{#S4.T1.12.22.10.7.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[34.68]{#S4.T1.12.22.10.8.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.951]{#S4.T1.12.22.10.9.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.038]{#S4.T1.12.22.10.10.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[33.53]{#S4.T1.12.22.10.11.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.979]{#S4.T1.12.22.10.12.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.012]{#S4.T1.12.22.10.13.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

Report issue for preceding element

[[Table 2]{#S4.T2.29.1.1 .ltx_text style="font-size:129%;"}: ]{.ltx_tag
.ltx_tag_table}[Novel view synthesis results Deblur-NeRF data set
variant in
 \[[40](https://arxiv.org/html/2403.13327v3#bib.bib40){.ltx_ref}\]. The
*Tanabata* scene has lower metrics due to an issue in the input data,
*cf*.[]{#S4.T2.30.2.3 .ltx_text} [[App.]{.ltx_text
.ltx_ref_tag} [0.B]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A2 "Appendix 0.B Data Sets and Metrics ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}.]{#S4.T2.30.2
.ltx_text style="font-size:129%;"}

::: {#S4.T2.15 .ltx_inline-block .ltx_align_center .ltx_transformed_outer style="width:429.5pt;height:114.3pt;vertical-align:-0.9pt;"}
[ ]{.ltx_transformed_inner
style="transform:translate(-23.9pt,6.3pt) scale(0.9,0.9) ;"}
:::

[Cozyroom]{#S4.T2.15.15.16.1.2.1 .ltx_text style="font-size:70%;"}

[Factory]{#S4.T2.15.15.16.1.3.1 .ltx_text style="font-size:70%;"}

[Pool]{#S4.T2.15.15.16.1.4.1 .ltx_text style="font-size:70%;"}

[Tanabata]{#S4.T2.15.15.16.1.5.1 .ltx_text style="font-size:70%;"}

[Trolley]{#S4.T2.15.15.16.1.6.1 .ltx_text style="font-size:70%;"}

[PSNR$\uparrow$]{#S4.T2.1.1.1.1.1 .ltx_text style="font-size:50%;"}

[ [[SSIM$\uparrow$]{#S4.T2.2.2.2.2.1.1.1 .ltx_text
style="font-size:50%;"}]{#S4.T2.2.2.2.2.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.2.2.2.2.1 .ltx_inline-block
.ltx_align_top}

[ [[LPIPS$\downarrow$]{#S4.T2.3.3.3.3.1.1.1 .ltx_text
style="font-size:50%;"}]{#S4.T2.3.3.3.3.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.3.3.3.3.1 .ltx_inline-block
.ltx_align_top}

[PSNR$\uparrow$]{#S4.T2.4.4.4.4.1 .ltx_text style="font-size:50%;"}

[ [[SSIM$\uparrow$]{#S4.T2.5.5.5.5.1.1.1 .ltx_text
style="font-size:50%;"}]{#S4.T2.5.5.5.5.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.5.5.5.5.1 .ltx_inline-block
.ltx_align_top}

[ [[LPIPS$\downarrow$]{#S4.T2.6.6.6.6.1.1.1 .ltx_text
style="font-size:50%;"}]{#S4.T2.6.6.6.6.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.6.6.6.6.1 .ltx_inline-block
.ltx_align_top}

[PSNR$\uparrow$]{#S4.T2.7.7.7.7.1 .ltx_text style="font-size:50%;"}

[ [[SSIM$\uparrow$]{#S4.T2.8.8.8.8.1.1.1 .ltx_text
style="font-size:50%;"}]{#S4.T2.8.8.8.8.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.8.8.8.8.1 .ltx_inline-block
.ltx_align_top}

[ [[LPIPS$\downarrow$]{#S4.T2.9.9.9.9.1.1.1 .ltx_text
style="font-size:50%;"}]{#S4.T2.9.9.9.9.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.9.9.9.9.1 .ltx_inline-block
.ltx_align_top}

[PSNR$\uparrow$]{#S4.T2.10.10.10.10.1 .ltx_text style="font-size:50%;"}

[ [[SSIM$\uparrow$]{#S4.T2.11.11.11.11.1.1.1 .ltx_text
style="font-size:50%;"}]{#S4.T2.11.11.11.11.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.11.11.11.11.1 .ltx_inline-block
.ltx_align_top}

[ [[LPIPS$\downarrow$]{#S4.T2.12.12.12.12.1.1.1 .ltx_text
style="font-size:50%;"}]{#S4.T2.12.12.12.12.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.12.12.12.12.1 .ltx_inline-block
.ltx_align_top}

[PSNR$\uparrow$]{#S4.T2.13.13.13.13.1 .ltx_text style="font-size:50%;"}

[ [[SSIM$\uparrow$]{#S4.T2.14.14.14.14.1.1.1 .ltx_text
style="font-size:50%;"}]{#S4.T2.14.14.14.14.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.14.14.14.14.1 .ltx_inline-block
.ltx_align_top}

[ [[LPIPS$\downarrow$]{#S4.T2.15.15.15.15.1.1.1 .ltx_text
style="font-size:50%;"}]{#S4.T2.15.15.15.15.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.15.15.1 .ltx_inline-block
.ltx_align_top}

[Splatfacto]{#S4.T2.15.15.17.1.1.1 .ltx_text style="font-size:70%;"}

[24.93]{#S4.T2.15.15.17.1.2.1 .ltx_text style="font-size:70%;"}

[ [[.802]{#S4.T2.15.15.17.1.3.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.17.1.3.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.17.1.3.1 .ltx_inline-block
.ltx_align_top}

[ [[.225]{#S4.T2.15.15.17.1.4.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.17.1.4.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.17.1.4.1 .ltx_inline-block
.ltx_align_top}

[21.28]{#S4.T2.15.15.17.1.5.1 .ltx_text style="font-size:70%;"}

[ [[.598]{#S4.T2.15.15.17.1.6.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.17.1.6.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.17.1.6.1 .ltx_inline-block
.ltx_align_top}

[ [[.440]{#S4.T2.15.15.17.1.7.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.17.1.7.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.17.1.7.1 .ltx_inline-block
.ltx_align_top}

[27.88]{#S4.T2.15.15.17.1.8.1 .ltx_text style="font-size:70%;"}

[ [[.763]{#S4.T2.15.15.17.1.9.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.17.1.9.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.17.1.9.1 .ltx_inline-block
.ltx_align_top}

[ [[.281]{#S4.T2.15.15.17.1.10.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.17.1.10.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.17.1.10.1 .ltx_inline-block
.ltx_align_top}

[18.52]{#S4.T2.15.15.17.1.11.1 .ltx_text style="font-size:70%;"}

[ [[.533]{#S4.T2.15.15.17.1.12.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.17.1.12.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.17.1.12.1 .ltx_inline-block
.ltx_align_top}

[ [[.433]{#S4.T2.15.15.17.1.13.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.17.1.13.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.17.1.13.1 .ltx_inline-block
.ltx_align_top}

[19.47]{#S4.T2.15.15.17.1.14.1 .ltx_text style="font-size:70%;"}

[ [[.564]{#S4.T2.15.15.17.1.15.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.17.1.15.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.17.1.15.1 .ltx_inline-block
.ltx_align_top}

[ [[.387]{#S4.T2.15.15.17.1.16.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.17.1.16.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.17.1.16.1 .ltx_inline-block
.ltx_align_top}

[MPR]{#S4.T2.15.15.18.2.1.1 .ltx_text
style="font-size:70%;"}[+]{#S4.T2.15.15.18.2.1.2 .ltx_text
style="font-size:50%;"}[Splatf.]{#S4.T2.15.15.18.2.1.3 .ltx_text
style="font-size:70%;"}

[29.26]{#S4.T2.15.15.18.2.2.1 .ltx_text style="font-size:70%;"}

[ [[.894]{#S4.T2.15.15.18.2.3.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.18.2.3.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.18.2.3.1 .ltx_inline-block
.ltx_align_top}

[ [[.093]{#S4.T2.15.15.18.2.4.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.18.2.4.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.18.2.4.1 .ltx_inline-block
.ltx_align_top}

[23.38]{#S4.T2.15.15.18.2.5.1 .ltx_text style="font-size:70%;"}

[ [[.737]{#S4.T2.15.15.18.2.6.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.18.2.6.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.18.2.6.1 .ltx_inline-block
.ltx_align_top}

[ [[.246]{#S4.T2.15.15.18.2.7.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.18.2.7.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.18.2.7.1 .ltx_inline-block
.ltx_align_top}

[30.96]{#S4.T2.15.15.18.2.8.1 .ltx_text style="font-size:70%;"}

[ [[.867]{#S4.T2.15.15.18.2.9.1.1.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}]{#S4.T2.15.15.18.2.9.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.18.2.9.1 .ltx_inline-block
.ltx_align_top}

[ [[.176]{#S4.T2.15.15.18.2.10.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.18.2.10.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.18.2.10.1 .ltx_inline-block
.ltx_align_top}

[22.77]{#S4.T2.15.15.18.2.11.1 .ltx_text style="font-size:70%;"}

[ [[.773]{#S4.T2.15.15.18.2.12.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.18.2.12.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.18.2.12.1 .ltx_inline-block
.ltx_align_top}

[ [[.227]{#S4.T2.15.15.18.2.13.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.18.2.13.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.18.2.13.1 .ltx_inline-block
.ltx_align_top}

[26.49]{#S4.T2.15.15.18.2.14.1 .ltx_text style="font-size:70%;"}

[ [[.854]{#S4.T2.15.15.18.2.15.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.18.2.15.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.18.2.15.1 .ltx_inline-block
.ltx_align_top}

[ [[.185]{#S4.T2.15.15.18.2.16.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.18.2.16.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.18.2.16.1 .ltx_inline-block
.ltx_align_top}

[Deblur-NeRF]{#S4.T2.15.15.19.3.1.1 .ltx_text style="font-size:70%;"}

[29.88]{#S4.T2.15.15.19.3.2.1 .ltx_text style="font-size:70%;"}

[ [[.890]{#S4.T2.15.15.19.3.3.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.19.3.3.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.19.3.3.1 .ltx_inline-block
.ltx_align_top}

[ [[.075]{#S4.T2.15.15.19.3.4.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.19.3.4.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.19.3.4.1 .ltx_inline-block
.ltx_align_top}

[26.06]{#S4.T2.15.15.19.3.5.1 .ltx_text style="font-size:70%;"}

[ [[.802]{#S4.T2.15.15.19.3.6.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.19.3.6.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.19.3.6.1 .ltx_inline-block
.ltx_align_top}

[ [[.211]{#S4.T2.15.15.19.3.7.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.19.3.7.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.19.3.7.1 .ltx_inline-block
.ltx_align_top}

[30.94]{#S4.T2.15.15.19.3.8.1 .ltx_text style="font-size:70%;"}

[ [[.840]{#S4.T2.15.15.19.3.9.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.19.3.9.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.19.3.9.1 .ltx_inline-block
.ltx_align_top}

[ [[.169]{#S4.T2.15.15.19.3.10.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.19.3.10.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.19.3.10.1 .ltx_inline-block
.ltx_align_top}

[22.56]{#S4.T2.15.15.19.3.11.1 .ltx_text style="font-size:70%;"}

[ [[.764]{#S4.T2.15.15.19.3.12.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.19.3.12.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.19.3.12.1 .ltx_inline-block
.ltx_align_top}

[ [[.229]{#S4.T2.15.15.19.3.13.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.19.3.13.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.19.3.13.1 .ltx_inline-block
.ltx_align_top}

[25.78]{#S4.T2.15.15.19.3.14.1 .ltx_text style="font-size:70%;"}

[ [[.812]{#S4.T2.15.15.19.3.15.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.19.3.15.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.19.3.15.1 .ltx_inline-block
.ltx_align_top}

[ [[.180]{#S4.T2.15.15.19.3.16.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.19.3.16.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.19.3.16.1 .ltx_inline-block
.ltx_align_top}

[BAD-NeRF]{#S4.T2.15.15.20.4.1.1 .ltx_text style="font-size:70%;"}

[30.97]{#S4.T2.15.15.20.4.2.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[ [[.901]{#S4.T2.15.15.20.4.3.1.1.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}]{#S4.T2.15.15.20.4.3.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.20.4.3.1 .ltx_inline-block
.ltx_align_top}

[ [[.055]{#S4.T2.15.15.20.4.4.1.1.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}]{#S4.T2.15.15.20.4.4.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.20.4.4.1 .ltx_inline-block
.ltx_align_top}

[31.65]{#S4.T2.15.15.20.4.5.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[ [[.904]{#S4.T2.15.15.20.4.6.1.1.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}]{#S4.T2.15.15.20.4.6.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.20.4.6.1 .ltx_inline-block
.ltx_align_top}

[ [[.123]{#S4.T2.15.15.20.4.7.1.1.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}]{#S4.T2.15.15.20.4.7.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.20.4.7.1 .ltx_inline-block
.ltx_align_top}

[31.72]{#S4.T2.15.15.20.4.8.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[ [[.858]{#S4.T2.15.15.20.4.9.1.1.1 .ltx_text
style="font-size:70%;"}]{#S4.T2.15.15.20.4.9.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.20.4.9.1 .ltx_inline-block
.ltx_align_top}

[ [[.115]{#S4.T2.15.15.20.4.10.1.1.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}]{#S4.T2.15.15.20.4.10.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.20.4.10.1 .ltx_inline-block
.ltx_align_top}

[23.82]{#S4.T2.15.15.20.4.11.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[ [[.831]{#S4.T2.15.15.20.4.12.1.1.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}]{#S4.T2.15.15.20.4.12.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.20.4.12.1 .ltx_inline-block
.ltx_align_top}

[ [[.138]{#S4.T2.15.15.20.4.13.1.1.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}]{#S4.T2.15.15.20.4.13.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.20.4.13.1 .ltx_inline-block
.ltx_align_top}

[28.25]{#S4.T2.15.15.20.4.14.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[ [[.873]{#S4.T2.15.15.20.4.15.1.1.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}]{#S4.T2.15.15.20.4.15.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.20.4.15.1 .ltx_inline-block
.ltx_align_top}

[ [[.091]{#S4.T2.15.15.20.4.16.1.1.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}]{#S4.T2.15.15.20.4.16.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.20.4.16.1 .ltx_inline-block
.ltx_align_top}

[Ours]{#S4.T2.15.15.21.5.1.1 .ltx_text style="font-size:70%;"}

[31.80]{#S4.T2.15.15.21.5.2.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[ [[.945]{#S4.T2.15.15.21.5.3.1.1.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}]{#S4.T2.15.15.21.5.3.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.21.5.3.1 .ltx_inline-block
.ltx_align_top}

[ [[.032]{#S4.T2.15.15.21.5.4.1.1.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}]{#S4.T2.15.15.21.5.4.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.21.5.4.1 .ltx_inline-block
.ltx_align_top}

[30.54]{#S4.T2.15.15.21.5.5.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[ [[.946]{#S4.T2.15.15.21.5.6.1.1.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}]{#S4.T2.15.15.21.5.6.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.21.5.6.1 .ltx_inline-block
.ltx_align_top}

[ [[.078]{#S4.T2.15.15.21.5.7.1.1.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}]{#S4.T2.15.15.21.5.7.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.21.5.7.1 .ltx_inline-block
.ltx_align_top}

[32.08]{#S4.T2.15.15.21.5.8.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[ [[.890]{#S4.T2.15.15.21.5.9.1.1.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}]{#S4.T2.15.15.21.5.9.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.21.5.9.1 .ltx_inline-block
.ltx_align_top}

[ [[.075]{#S4.T2.15.15.21.5.10.1.1.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}]{#S4.T2.15.15.21.5.10.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.21.5.10.1 .ltx_inline-block
.ltx_align_top}

[24.79]{#S4.T2.15.15.21.5.11.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[ [[.912]{#S4.T2.15.15.21.5.12.1.1.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}]{#S4.T2.15.15.21.5.12.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.21.5.12.1 .ltx_inline-block
.ltx_align_top}

[ [[.079]{#S4.T2.15.15.21.5.13.1.1.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}]{#S4.T2.15.15.21.5.13.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.21.5.13.1 .ltx_inline-block
.ltx_align_top}

[30.16]{#S4.T2.15.15.21.5.14.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[ [[.933]{#S4.T2.15.15.21.5.15.1.1.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}]{#S4.T2.15.15.21.5.15.1.1 .ltx_p
style="width:14.7pt;"} ]{#S4.T2.15.15.21.5.15.1 .ltx_inline-block
.ltx_align_top}

[ [[.044]{#S4.T2.15.15.21.5.16.1.1.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}]{#S4.T2.15.15.21.5.16.1.1 .ltx_p
style="width:17.5pt;"} ]{#S4.T2.15.15.21.5.16.1 .ltx_inline-block
.ltx_align_top}

Report issue for preceding element

::: {#S4.SS1 .section .ltx_subsection}
### [4.1 ]{.ltx_tag .ltx_tag_subsection}Synthetic Data {#synthetic-data .ltx_title .ltx_title_subsection}

Report issue for preceding element

::: {#S4.SS1.p1 .ltx_para}
3DGS uses a SfM-based sparse point cloud as seed points for the
initialization of the Gaussian means and scales, which is typically
obtained from
COLMAP \[[31](https://arxiv.org/html/2403.13327v3#bib.bib31){.ltx_ref}\].
However, with significantly blurry and other otherwise noisy data,
COLMAP often fails which prevents us from obtaining good SfM point cloud
for initialization on very blurry synthetic data sets like Deblur-NeRF.
To overcome this, we compute an initial point cloud using
SIFT \[[24](https://arxiv.org/html/2403.13327v3#bib.bib24){.ltx_ref}\]
feature matching and triangulation with the COLMAP-estimated poses.

Report issue for preceding element
:::

::: {#S4.SS1.SSS0.Px1 .section .ltx_paragraph}
##### BAD-NeRF data set variant {#bad-nerf-data-set-variant .ltx_title .ltx_title_paragraph}

Report issue for preceding element

::: {#S4.SS1.SSS0.Px1.p1 .ltx_para}
To compare our performance with existing work, we use the BAD-NeRF
re-render \[[40](https://arxiv.org/html/2403.13327v3#bib.bib40){.ltx_ref}\]
of the synthetic Deblur-NeRF data
set \[[25](https://arxiv.org/html/2403.13327v3#bib.bib25){.ltx_ref}\].
We computed the results of our method, the Splatfacto baseline and a
variant where, prior to COLMAP registration, the images were deblurred
with
MPR \[[50](https://arxiv.org/html/2403.13327v3#bib.bib50){.ltx_ref}\].
Similarly
to \[[40](https://arxiv.org/html/2403.13327v3#bib.bib40){.ltx_ref}\], we
only use the blurry images for pose registration in COLMAP. The results
of the experiment are presented in [[Table]{.ltx_text
.ltx_ref_tag} [2]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S4.T2 "In 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref},
where they are also compared to the previous NeRF-based methods as
reported
in \[[40](https://arxiv.org/html/2403.13327v3#bib.bib40){.ltx_ref}\],
where we also note that the original
Deblur-NeRF \[[25](https://arxiv.org/html/2403.13327v3#bib.bib25){.ltx_ref}\]
has been evaluated using ground truth training poses. Our method clearly
outperforms all baselines.

Report issue for preceding element
:::
:::

::: {#S4.SS1.SSS0.Px2 .section .ltx_paragraph}
##### Re-rendered data {#re-rendered-data .ltx_title .ltx_title_paragraph}

Report issue for preceding element

::: {#S4.SS1.SSS0.Px2.p1 .ltx_para}
To study the individual components of our method in more detail, we
rendered new variants of the Deblur-NeRF data set to simulate motion
blur, rolling shutter, and noisy pose effects respectively (see
[[App.]{.ltx_text .ltx_ref_tag} [0.B]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A2 "Appendix 0.B Data Sets and Metrics ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
for details). We experiment on the following synthetic data set
variants: [(i)]{#S4.SS1.SSS0.Px2.p1.1.1 .ltx_text .ltx_font_italic} a
motion blur variant, which closely matches the motion blur variant of
the original Deblur-NeRF data set, [(ii)]{#S4.SS1.SSS0.Px2.p1.1.2
.ltx_text .ltx_font_italic} a rolling shutter only variant,
[(iii)]{#S4.SS1.SSS0.Px2.p1.1.3 .ltx_text .ltx_font_italic}  and a
translational and angular pose noise variant, without RS or motion blur
effects. In the motion blur and rolling shutter variants, we use the
synthetic ground truth velocities $(v_{j},\omega_{j})$ without
additional pose noise. For these variants, we keep poses and velocities
fixed during optimization. In this experiment, the ground truth poses
were also used for initial point cloud triangulation instead of
COLMAP-estimated poses.

Report issue for preceding element
:::

::: {#S4.SS1.SSS0.Px2.p2 .ltx_para}
We train 3DGS models for all of the above synthetic data sets using 3DGS
Splatfacto method as our baseline and we compare novel-view synthesis
metrics with our method with the appropriate compensation mode enabled
(*i.e*.[]{#S4.SS1.SSS0.Px2.p2.1.2 .ltx_text}, rolling shutter
compensation but no motion blur compensation or pose optimization for RS
case). The results are shown in [[Table]{.ltx_text
.ltx_ref_tag} [1]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S4.T1 "In 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
and qualitatively visualized in [[Figs.]{.ltx_text
.ltx_ref_tag} [1]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S1.F1 "In 1 Introduction ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
and [[2]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}.
Our method consistently outperforms the 3DGS baseline across all
scenarios, indicating the effectiveness of our approach in compensating
for blurring and RS effects arising from camera motion in pixel space.

Report issue for preceding element
:::

[[Table 3]{#S4.T3.45.1.1 .ltx_text style="font-size:129%;"}: ]{.ltx_tag
.ltx_tag_table}[PSNR metric ablation study and comparison for smartphone
data.]{#S4.T3.46.2 .ltx_text style="font-size:129%;"}

::: {#S4.T3.41 .ltx_inline-block .ltx_align_center .ltx_transformed_outer style="width:253.4pt;height:293.9pt;vertical-align:-0.0pt;"}
[ ]{.ltx_transformed_inner
style="transform:translate(-14.1pt,16.3pt) scale(0.9,0.9) ;"}

  --------------------------------------------------------------------------------------------- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                                                [ [[Splatf.]{#S4.T3.5.5.5.7.1.1.1 .ltx_text style="font-size:70%;"}]{#S4.T3.5.5.5.7.1.1 .ltx_p style="width:21.0pt;"} ]{#S4.T3.5.5.5.7.1 .ltx_inline-block .ltx_align_top}   [ [[]{#S4.T3.1.1.1.1.1.1.1 .ltx_text style="font-size:70%;"}$\smallsetminus$[MB]{#S4.T3.1.1.1.1.1.1.2 .ltx_text style="font-size:70%;"}]{#S4.T3.1.1.1.1.1.1 .ltx_p style="width:21.0pt;"} ]{#S4.T3.1.1.1.1.1 .ltx_inline-block .ltx_align_top}   [ [[]{#S4.T3.2.2.2.2.1.1.1 .ltx_text style="font-size:70%;"}$\smallsetminus$[RS]{#S4.T3.2.2.2.2.1.1.2 .ltx_text style="font-size:70%;"}]{#S4.T3.2.2.2.2.1.1 .ltx_p style="width:21.0pt;"} ]{#S4.T3.2.2.2.2.1 .ltx_inline-block .ltx_align_top}   [ [$\smallsetminus$[P.opt.]{#S4.T3.3.3.3.3.1.1.1 .ltx_text style="font-size:70%;"}]{#S4.T3.3.3.3.3.1.1 .ltx_p style="width:21.0pt;"} ]{#S4.T3.3.3.3.3.1 .ltx_inline-block .ltx_align_top}   [ [$\smallsetminus$[V.opt.]{#S4.T3.4.4.4.4.1.1.1 .ltx_text style="font-size:70%;"}]{#S4.T3.4.4.4.4.1.1 .ltx_p style="width:21.0pt;"} ]{#S4.T3.4.4.4.4.1 .ltx_inline-block .ltx_align_top}   [ [[]{#S4.T3.5.5.5.5.1.1.1 .ltx_text style="font-size:70%;"}$\smallsetminus$[VIO]{#S4.T3.5.5.5.5.1.1.2 .ltx_text style="font-size:70%;"}]{#S4.T3.5.5.5.5.1.1 .ltx_p style="width:21.0pt;"} ]{#S4.T3.5.5.5.5.1 .ltx_inline-block .ltx_align_top}   [ [[CVR]{#S4.T3.5.5.5.8.1.1.1 .ltx_text style="font-size:70%;"}]{#S4.T3.5.5.5.8.1.1 .ltx_p style="width:21.0pt;"} ]{#S4.T3.5.5.5.8.1 .ltx_inline-block .ltx_align_top}   [ [[Ours]{#S4.T3.5.5.5.9.1.1.1 .ltx_text style="font-size:70%;"}]{#S4.T3.5.5.5.9.1.1 .ltx_p style="width:21.0pt;"} ]{#S4.T3.5.5.5.9.1 .ltx_inline-block .ltx_align_top}
  [Motion blur]{#S4.T3.13.13.13.9.1 .ltx_text style="font-size:70%;"}                           $-$                                                                                                                                                                          $-$                                                                                                                                                                                                                                              $✓$                                                                                                                                                                                                                                              $✓$                                                                                                                                                                                         $✓$                                                                                                                                                                                         $✓$                                                                                                                                                                                                                                               $✓$                                                                                                                                                                      $✓$
  [Rolling shut.]{#S4.T3.20.20.20.8.1 .ltx_text style="font-size:70%;"}                         $-$                                                                                                                                                                          $✓$                                                                                                                                                                                                                                              $-$                                                                                                                                                                                                                                              $✓$                                                                                                                                                                                         $✓$                                                                                                                                                                                         $✓$                                                                                                                                                                                                                                               [CVR]{#S4.T3.20.20.20.9.1 .ltx_text style="font-size:70%;"}                                                                                                              $✓$
  [Pose opt.]{#S4.T3.28.28.28.9.1 .ltx_text style="font-size:70%;"}                             $-$                                                                                                                                                                          $✓$                                                                                                                                                                                                                                              $✓$                                                                                                                                                                                                                                              $-$                                                                                                                                                                                         $✓$                                                                                                                                                                                         $✓$                                                                                                                                                                                                                                               $✓$                                                                                                                                                                      $✓$
  [Velocity opt.]{#S4.T3.36.36.36.9.1 .ltx_text style="font-size:70%;"}                         $-$                                                                                                                                                                          $✓$                                                                                                                                                                                                                                              $✓$                                                                                                                                                                                                                                              $✓$                                                                                                                                                                                         $-$                                                                                                                                                                                         $✓$                                                                                                                                                                                                                                               $✓$                                                                                                                                                                      $✓$
  [VIO vel. init.]{#S4.T3.41.41.41.6.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                                     $✓$                                                                                                                                                                                                                                              $✓$                                                                                                                                                                                                                                              $✓$                                                                                                                                                                                                                                                                                                                                                                                     $-$                                                                                                                                                                                                                                                                                                                                                                                                                        $✓$
  [iphone-lego1]{#S4.T3.41.41.42.1.1.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}    [28.05]{#S4.T3.41.41.42.1.2.1 .ltx_text style="font-size:70%;"}                                                                                                              [28.12]{#S4.T3.41.41.42.1.3.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                  [29.20]{#S4.T3.41.41.42.1.4.1 .ltx_text .ltx_font_italic style="font-size:70%;"}                                                                                                                                                                 [28.59]{#S4.T3.41.41.42.1.5.1 .ltx_text style="font-size:70%;"}                                                                                                                             [28.71]{#S4.T3.41.41.42.1.6.1 .ltx_text style="font-size:70%;"}                                                                                                                             [29.03]{#S4.T3.41.41.42.1.7.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                   [23.26]{#S4.T3.41.41.42.1.8.1 .ltx_text style="font-size:70%;"}                                                                                                          [29.20]{#S4.T3.41.41.42.1.9.1 .ltx_text .ltx_font_bold style="font-size:70%;"}
  [iphone-lego2]{#S4.T3.41.41.43.2.1.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}    [27.85]{#S4.T3.41.41.43.2.2.1 .ltx_text style="font-size:70%;"}                                                                                                              [27.88]{#S4.T3.41.41.43.2.3.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                  [27.95]{#S4.T3.41.41.43.2.4.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                  [27.39]{#S4.T3.41.41.43.2.5.1 .ltx_text style="font-size:70%;"}                                                                                                                             [28.15]{#S4.T3.41.41.43.2.6.1 .ltx_text .ltx_font_italic style="font-size:70%;"}                                                                                                            [28.55]{#S4.T3.41.41.43.2.7.1 .ltx_text .ltx_font_bold style="font-size:70%;"}                                                                                                                                                                    [26.45]{#S4.T3.41.41.43.2.8.1 .ltx_text style="font-size:70%;"}                                                                                                          [27.95]{#S4.T3.41.41.43.2.9.1 .ltx_text style="font-size:70%;"}
  [iphone-lego3]{#S4.T3.41.41.44.3.1.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}    [23.75]{#S4.T3.41.41.44.3.2.1 .ltx_text style="font-size:70%;"}                                                                                                              [23.71]{#S4.T3.41.41.44.3.3.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                  [24.50]{#S4.T3.41.41.44.3.4.1 .ltx_text .ltx_font_italic style="font-size:70%;"}                                                                                                                                                                 [24.10]{#S4.T3.41.41.44.3.5.1 .ltx_text style="font-size:70%;"}                                                                                                                             [24.22]{#S4.T3.41.41.44.3.6.1 .ltx_text style="font-size:70%;"}                                                                                                                             [23.78]{#S4.T3.41.41.44.3.7.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                   [22.45]{#S4.T3.41.41.44.3.8.1 .ltx_text style="font-size:70%;"}                                                                                                          [24.50]{#S4.T3.41.41.44.3.9.1 .ltx_text .ltx_font_bold style="font-size:70%;"}
  [iphone-pots1]{#S4.T3.41.41.45.4.1.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}    [28.32]{#S4.T3.41.41.45.4.2.1 .ltx_text style="font-size:70%;"}                                                                                                              [28.58]{#S4.T3.41.41.45.4.3.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                  [29.10]{#S4.T3.41.41.45.4.4.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                  [28.91]{#S4.T3.41.41.45.4.5.1 .ltx_text style="font-size:70%;"}                                                                                                                             [28.93]{#S4.T3.41.41.45.4.6.1 .ltx_text style="font-size:70%;"}                                                                                                                             [29.18]{#S4.T3.41.41.45.4.7.1 .ltx_text .ltx_font_bold style="font-size:70%;"}                                                                                                                                                                    [24.44]{#S4.T3.41.41.45.4.8.1 .ltx_text style="font-size:70%;"}                                                                                                          [29.10]{#S4.T3.41.41.45.4.9.1 .ltx_text .ltx_font_italic style="font-size:70%;"}
  [iphone-pots2]{#S4.T3.41.41.46.5.1.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}    [27.25]{#S4.T3.41.41.46.5.2.1 .ltx_text style="font-size:70%;"}                                                                                                              [27.39]{#S4.T3.41.41.46.5.3.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                  [28.00]{#S4.T3.41.41.46.5.4.1 .ltx_text .ltx_font_italic style="font-size:70%;"}                                                                                                                                                                 [27.68]{#S4.T3.41.41.46.5.5.1 .ltx_text style="font-size:70%;"}                                                                                                                             [26.66]{#S4.T3.41.41.46.5.6.1 .ltx_text style="font-size:70%;"}                                                                                                                             [27.81]{#S4.T3.41.41.46.5.7.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                   [23.64]{#S4.T3.41.41.46.5.8.1 .ltx_text style="font-size:70%;"}                                                                                                          [28.00]{#S4.T3.41.41.46.5.9.1 .ltx_text .ltx_font_bold style="font-size:70%;"}
  [pixel5-lamp]{#S4.T3.41.41.47.6.1.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}     [28.22]{#S4.T3.41.41.47.6.2.1 .ltx_text style="font-size:70%;"}                                                                                                              [30.91]{#S4.T3.41.41.47.6.3.1 .ltx_text .ltx_font_italic style="font-size:70%;"}                                                                                                                                                                 [28.38]{#S4.T3.41.41.47.6.4.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                  [28.77]{#S4.T3.41.41.47.6.5.1 .ltx_text style="font-size:70%;"}                                                                                                                             [29.75]{#S4.T3.41.41.47.6.6.1 .ltx_text style="font-size:70%;"}                                                                                                                             [29.70]{#S4.T3.41.41.47.6.7.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                   [31.95]{#S4.T3.41.41.47.6.8.1 .ltx_text .ltx_font_bold style="font-size:70%;"}                                                                                           [30.46]{#S4.T3.41.41.47.6.9.1 .ltx_text style="font-size:70%;"}
  [pixel5-plant]{#S4.T3.41.41.48.7.1.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}    [26.57]{#S4.T3.41.41.48.7.2.1 .ltx_text style="font-size:70%;"}                                                                                                              [27.41]{#S4.T3.41.41.48.7.3.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                  [27.45]{#S4.T3.41.41.48.7.4.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                  [26.81]{#S4.T3.41.41.48.7.5.1 .ltx_text style="font-size:70%;"}                                                                                                                             [27.37]{#S4.T3.41.41.48.7.6.1 .ltx_text style="font-size:70%;"}                                                                                                                             [27.69]{#S4.T3.41.41.48.7.7.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                   [28.20]{#S4.T3.41.41.48.7.8.1 .ltx_text .ltx_font_bold style="font-size:70%;"}                                                                                           [27.90]{#S4.T3.41.41.48.7.9.1 .ltx_text .ltx_font_italic style="font-size:70%;"}
  [pixel5-table]{#S4.T3.41.41.49.8.1.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}    [28.93]{#S4.T3.41.41.49.8.2.1 .ltx_text style="font-size:70%;"}                                                                                                              [30.82]{#S4.T3.41.41.49.8.3.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                  [29.25]{#S4.T3.41.41.49.8.4.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                  [30.33]{#S4.T3.41.41.49.8.5.1 .ltx_text style="font-size:70%;"}                                                                                                                             [31.16]{#S4.T3.41.41.49.8.6.1 .ltx_text style="font-size:70%;"}                                                                                                                             [31.47]{#S4.T3.41.41.49.8.7.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                   [32.42]{#S4.T3.41.41.49.8.8.1 .ltx_text .ltx_font_bold style="font-size:70%;"}                                                                                           [31.86]{#S4.T3.41.41.49.8.9.1 .ltx_text .ltx_font_italic style="font-size:70%;"}
  [s20-bike]{#S4.T3.41.41.50.9.1.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}        [27.35]{#S4.T3.41.41.50.9.2.1 .ltx_text style="font-size:70%;"}                                                                                                              [27.74]{#S4.T3.41.41.50.9.3.1 .ltx_text .ltx_font_italic style="font-size:70%;"}                                                                                                                                                                 [27.57]{#S4.T3.41.41.50.9.4.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                  [27.58]{#S4.T3.41.41.50.9.5.1 .ltx_text style="font-size:70%;"}                                                                                                                             [27.58]{#S4.T3.41.41.50.9.6.1 .ltx_text style="font-size:70%;"}                                                                                                                             [27.72]{#S4.T3.41.41.50.9.7.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                   [26.62]{#S4.T3.41.41.50.9.8.1 .ltx_text style="font-size:70%;"}                                                                                                          [28.93]{#S4.T3.41.41.50.9.9.1 .ltx_text .ltx_font_bold style="font-size:70%;"}
  [s20-bikerack]{#S4.T3.41.41.51.10.1.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}   [25.98]{#S4.T3.41.41.51.10.2.1 .ltx_text style="font-size:70%;"}                                                                                                             [29.39]{#S4.T3.41.41.51.10.3.1 .ltx_text .ltx_font_italic style="font-size:70%;"}                                                                                                                                                                [27.92]{#S4.T3.41.41.51.10.4.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                 [26.23]{#S4.T3.41.41.51.10.5.1 .ltx_text style="font-size:70%;"}                                                                                                                            [26.09]{#S4.T3.41.41.51.10.6.1 .ltx_text style="font-size:70%;"}                                                                                                                            [28.92]{#S4.T3.41.41.51.10.7.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                  [27.77]{#S4.T3.41.41.51.10.8.1 .ltx_text style="font-size:70%;"}                                                                                                         [29.74]{#S4.T3.41.41.51.10.9.1 .ltx_text .ltx_font_bold style="font-size:70%;"}
  [s20-sign]{#S4.T3.41.41.52.11.1.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}       [23.71]{#S4.T3.41.41.52.11.2.1 .ltx_text style="font-size:70%;"}                                                                                                             [25.93]{#S4.T3.41.41.52.11.3.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                 [24.47]{#S4.T3.41.41.52.11.4.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                 [25.44]{#S4.T3.41.41.52.11.5.1 .ltx_text style="font-size:70%;"}                                                                                                                            [25.54]{#S4.T3.41.41.52.11.6.1 .ltx_text style="font-size:70%;"}                                                                                                                            [26.28]{#S4.T3.41.41.52.11.7.1 .ltx_text .ltx_font_italic style="font-size:70%;"}                                                                                                                                                                 [24.19]{#S4.T3.41.41.52.11.8.1 .ltx_text style="font-size:70%;"}                                                                                                         [26.84]{#S4.T3.41.41.52.11.9.1 .ltx_text .ltx_font_bold style="font-size:70%;"}
  [average]{#S4.T3.41.41.53.12.1.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}        [26.91]{#S4.T3.41.41.53.12.2.1 .ltx_text style="font-size:70%;"}                                                                                                             [27.99]{#S4.T3.41.41.53.12.3.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                 [27.62]{#S4.T3.41.41.53.12.4.1 .ltx_text style="font-size:70%;"}                                                                                                                                                                                 [27.44]{#S4.T3.41.41.53.12.5.1 .ltx_text style="font-size:70%;"}                                                                                                                            [27.65]{#S4.T3.41.41.53.12.6.1 .ltx_text style="font-size:70%;"}                                                                                                                            [28.19]{#S4.T3.41.41.53.12.7.1 .ltx_text .ltx_font_italic style="font-size:70%;"}                                                                                                                                                                 [26.49]{#S4.T3.41.41.53.12.8.1 .ltx_text style="font-size:70%;"}                                                                                                         [28.59]{#S4.T3.41.41.53.12.9.1 .ltx_text .ltx_font_bold style="font-size:70%;"}
  --------------------------------------------------------------------------------------------- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
:::

Report issue for preceding element

![[[Figure 4]{#S4.F4.18.3.1 .ltx_text style="font-size:129%;"}:
]{.ltx_tag .ltx_tag_figure}[Real data examples with scenes captured with
smartphones. From top to bottom: Using COLMAP poses without motion blur
compensation (baseline); with motion blur compensation (ours); reference
evaluation image; $l_{2}$ error contributions, ours (red: 30%, yellow:
60%, white: 10%); $l_{2}$ error differences ours vs baseline (red: more
error, blue: less error). Scenes: [lego1]{#S4.F4.4.2.1 .ltx_text
.ltx_font_typewriter} (iPhone), [pots2]{#S4.F4.4.2.2 .ltx_text
.ltx_font_typewriter} (iPhone), [table]{#S4.F4.4.2.3 .ltx_text
.ltx_font_typewriter} (Pixel), [bike]{#S4.F4.4.2.4 .ltx_text
.ltx_font_typewriter} (S20).]{#S4.F4.4.2 .ltx_text
style="font-size:129%;"}](3dgs_deblur_paper_files/colmap-sai-cli-vels-blur-scored-iphone-lego1-baseline_JJYT.jpg){#S4.F4.pic1.3.3.3.3.3.3.3.3.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.1.g1
.ltx_graphics .ltx_img_landscape width="144" height="108"}

Report issue for preceding element

![[[Figure 5]{#S4.F5.4.1.1 .ltx_text style="font-size:129%;"}:
]{.ltx_tag .ltx_tag_figure}[Empirical ablations on deblurring with and
without rolling shutter pose optimization on four real-world smartphone
data scenes with rolling shutter effects. Our method with both motion
blur and rolling shutter compensation (MBRS) gives sharper artefact-free
reconstructions compared to just motion blur compensation
(MB).]{#S4.F5.5.2 .ltx_text
style="font-size:129%;"}](3dgs_deblur_paper_files/colmap-sai-cli-vels-blur-scored-pixel5-table-rsabl-base_JJYT.jpg){#S4.F5.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.g1
.ltx_graphics .ltx_img_landscape width="144" height="108"}

Report issue for preceding element
:::
:::

::: {#S4.SS2 .section .ltx_subsection}
### [4.2 ]{.ltx_tag .ltx_tag_subsection}Smartphone Data {#smartphone-data .ltx_title .ltx_title_subsection}

Report issue for preceding element

::: {#S4.SS2.p1 .ltx_para}
To the end of real data evaluation, we use a new data set recorded using
three different smartphones: Samsung S20 FE, Google Pixel 5 and iPhone
15 Pro. The first two are Android phones with a known, and relatively
large rolling-shutter readout time $T_{ro}$.

Report issue for preceding element
:::

::: {#S4.SS2.p2 .ltx_para}
The data set consists of 11 short handheld recordings of various scenes
collected using the Spectacular Rec
application \[[1](https://arxiv.org/html/2403.13327v3#bib.bib1){.ltx_ref}\],
which records synchronized IMU and video data, together with the
built-in (factory) calibration information from each device. The same
application is available both for Android and iOS, and it was used to
capture the raw image and IMU data on the devices.

Report issue for preceding element
:::

::: {#S4.SS2.SSS0.Px1 .section .ltx_paragraph}
##### Preprocessing and VIO velocity estimation {#preprocessing-and-vio-velocity-estimation .ltx_title .ltx_title_paragraph}

Report issue for preceding element

::: {#S4.SS2.SSS0.Px1.p1 .ltx_para}
We first process the recorded data with the Spectacular AI
SDK \[[1](https://arxiv.org/html/2403.13327v3#bib.bib1){.ltx_ref}\], to
obtain the following: [(i)]{#S4.SS2.SSS0.Px1.p1.2.1 .ltx_text
.ltx_font_italic} A sparse set of key frames with minimum distance of
10 cm, selected to approximately minimize motion blur among the set of
candidate key frames (see [[Sec.]{.ltx_text
.ltx_ref_tag} [0.A.5]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A1.SS5 "0.A.5 Key Frame Selection ‣ Appendix 0.A Method Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
for details). [(ii)]{#S4.SS2.SSS0.Px1.p1.2.2 .ltx_text
.ltx_font_italic} For each key frame, the initial estimates for the
frame velocities $(v_{j}^{VIO},\omega_{j}^{VIO})$, based on the fusion
of IMU and video data. [(iii)]{#S4.SS2.SSS0.Px1.p1.2.3 .ltx_text
.ltx_font_italic} Approximate VISLAM-based poses $P_{j}^{SAI}$.

Report issue for preceding element
:::
:::

::: {#S4.SS2.SSS0.Px2 .section .ltx_paragraph}
##### Training and evaluation split {#training-and-evaluation-split .ltx_title .ltx_title_paragraph}

Report issue for preceding element

::: {#S4.SS2.SSS0.Px2.p1 .ltx_para}
As with the synthetic data and in
\[[25](https://arxiv.org/html/2403.13327v3#bib.bib25){.ltx_ref}\], we
aim to select the least blurry frames for evaluation, which is performed
by splitting the (ordered) key frames to subsets of eight consecutive
key frames, and for each subset, picking the least blurry one for
evaluation. We use the same motion blur metric as for key frame
selection in [[Sec.]{.ltx_text .ltx_ref_tag} [0.A.5]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A1.SS5 "0.A.5 Key Frame Selection ‣ Appendix 0.A Method Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}.

Report issue for preceding element
:::
:::

::: {#S4.SS2.SSS0.Px3 .section .ltx_paragraph}
##### Pose and intrinsic estimation {#pose-and-intrinsic-estimation .ltx_title .ltx_title_paragraph}

Report issue for preceding element

::: {#S4.SS2.SSS0.Px3.p1 .ltx_para}
After preprocessing, the poses of the key frames and camera intrinsics
are estimated using
COLMAP \[[31](https://arxiv.org/html/2403.13327v3#bib.bib31){.ltx_ref}\]
(through Nerfstudio). In our main results, we only included sequences in
which COLMAP did not fail due to excessively difficult visual
conditions. In [[App.]{.ltx_text .ltx_ref_tag} [0.D]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A4 "Appendix 0.D Additional Results ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
we present additional experiments where manual calibration results or
built-in calibration data from the devices is used in place of COLMAP
intrinsics.

Report issue for preceding element
:::
:::

::: {#S4.SS2.SSS0.Px4 .section .ltx_paragraph}
##### COLMAP baseline {#colmap-baseline .ltx_title .ltx_title_paragraph}

Report issue for preceding element

::: {#S4.SS2.SSS0.Px4.p1 .ltx_para}
Our baseline comparison involves the Splatfacto method paired with poses
estimated by COLMAP, without any specific compensation for motion blur
or rolling shutter effects. This baseline serves as a reference point
for evaluating the efficacy of our proposed motion blur compensation
method.

Report issue for preceding element
:::
:::

::: {#S4.SS2.SSS0.Px5 .section .ltx_paragraph}
##### CVR de-rolling baseline {#cvr-de-rolling-baseline .ltx_title .ltx_title_paragraph}

Report issue for preceding element

::: {#S4.SS2.SSS0.Px5.p1 .ltx_para}
As a baseline for rolling shutter compensation, we experimented with a
variant where, prior to the COLMAP phase, the image data was de-rolled
with
CVR \[[6](https://arxiv.org/html/2403.13327v3#bib.bib6){.ltx_ref}\]. For
a fair comparison, we enabled the other optimization features in our
method in this test. We also note that this test was omitted with
synthetic data as CVR requires access to consecutive frames, which were
not available in the sparse synthetic test.

Report issue for preceding element
:::
:::

::: {#S4.SS2.SSS0.Px6 .section .ltx_paragraph}
##### Ablation study {#ablation-study .ltx_title .ltx_title_paragraph}

Report issue for preceding element

::: {#S4.SS2.SSS0.Px6.p1 .ltx_para}
We also performed an ablation study where each of the main components of
the method are individually disabled: [(i)]{#S4.SS2.SSS0.Px6.p1.7.1
.ltx_text .ltx_font_italic} motion blur compensation, by setting
$T_{e} = 0$, $N_{blur} = 1$; [(ii)]{#S4.SS2.SSS0.Px6.p1.7.2 .ltx_text
.ltx_font_italic} rolling shutter compensation by setting $T_{ro} = 0$;
[(iii)]{#S4.SS2.SSS0.Px6.p1.7.3 .ltx_text .ltx_font_italic} pose
optimization; [(iv)]{#S4.SS2.SSS0.Px6.p1.7.4 .ltx_text
.ltx_font_italic} pptimization of linear and angular velocities
$(v_{i},\omega_{i})$; [(v)]{#S4.SS2.SSS0.Px6.p1.7.5 .ltx_text
.ltx_font_italic} velocity optimization initialization from VIO by
setting $v_{i} = \omega_{i} = 0$ instead of $v_{i} = v_{i}^{VIO}$,
$\omega_{i} = \omega_{i}^{VIO}$.

Report issue for preceding element
:::
:::

::: {#S4.SS2.SSS0.Px7 .section .ltx_paragraph}
##### Results {#results .ltx_title .ltx_title_paragraph}

Report issue for preceding element

::: {#S4.SS2.SSS0.Px7.p1 .ltx_para}
The real data results are given in [[Table]{.ltx_text
.ltx_ref_tag} [3]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S4.T3 "In Re-rendered data ‣ 4.1 Synthetic Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}.
All features included in the ablation study display a positive impact
and our method achieves the best overall performance in therms of the
PSNR metric. Performance compared to CVR pre-processing is mixed: while
CVR combined with our other compensation features performs best on the
data from Pixel 5, it is worse on S20 and clearly worse if enabled for
iPhone (with low rolling shutter readout time). Qualitatively we noticed
that CVR outputs are often contaminated by various types of artefacts
that cause bad performance in 3DGS, and conclude that our method is more
robust than CVR for rolling-shutter compensation for static scene
reconstructions.

Report issue for preceding element
:::

::: {#S4.SS2.SSS0.Px7.p2 .ltx_para}
Qualitatively, the sharpness of the reconstructions is subtly, but
noticeably increased when motion blur compensation is enabled, as
demonstrated in the highlighted details in [[Figs.]{.ltx_text
.ltx_ref_tag} [4]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S4.F4 "In Re-rendered data ‣ 4.1 Synthetic Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
and [[8]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A4.F8 "Figure 8 ‣ Blur-based key frame selection ‣ 0.D.0.1 Alternative intrinsics ‣ Appendix 0.D Additional Results ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}.
Furthermore, [[Fig.]{.ltx_text .ltx_ref_tag} [5]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S4.F5 "In Re-rendered data ‣ 4.1 Synthetic Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
shows that rolling shutter compensation also noticeably improves the
reconstruction quality, especially closer to the edges of the evaluation
images.

Report issue for preceding element
:::

::: {#S4.SS2.SSS0.Px7.p3 .ltx_para}
The automatic selection of key frames for evaluation also played a
significant role in our analysis, particularly in highlighting areas
poorly represented in the training data. These areas, especially near
the edges of the visual field, were more susceptible to artefacts that
disproportionately affected the PSNR metric, underlining the challenges
in balancing motion blur compensation with the preservation of image
quality across the entire scene. These effects are visible near the
edges of the error metric figures in [[Figs.]{.ltx_text
.ltx_ref_tag} [4]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S4.F4 "In Re-rendered data ‣ 4.1 Synthetic Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
and [[8]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A4.F8 "Figure 8 ‣ Blur-based key frame selection ‣ 0.D.0.1 Alternative intrinsics ‣ Appendix 0.D Additional Results ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}.

Report issue for preceding element
:::

[[Table 4]{#S4.T4.20.2.1 .ltx_text style="font-size:129%;"}: ]{.ltx_tag
.ltx_tag_table}[Real data timing results. Training wall clock time $T$
(minutes) in the baseline method and ours with different features
disabled.]{#S4.T4.2.1 .ltx_text style="font-size:129%;"}

                                                                                                                                                                                                                                                                                 [Splatfacto]{#S4.T4.4.2.6.1 .ltx_text style="font-size:70%;"}   []{#S4.T4.3.1.1.1 .ltx_text style="font-size:70%;"}$\smallsetminus$[MB]{#S4.T4.3.1.1.2 .ltx_text style="font-size:70%;"}   []{#S4.T4.4.2.2.1 .ltx_text style="font-size:70%;"}$\smallsetminus$[RS]{#S4.T4.4.2.2.2 .ltx_text style="font-size:70%;"}   [Ours]{#S4.T4.4.2.7.1 .ltx_text style="font-size:70%;"}
  --------------------------------------------------------------------------------------- --------------------------------------------------------- ---------------------------------------------------------------------------------------------------------------------------- --------------------------------------------------------------- -------------------------------------------------------------------------------------------------------------------------- -------------------------------------------------------------------------------------------------------------------------- -----------------------------------------------------------
  [iphone-lego1]{#S4.T4.5.3.2.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}     [21]{#S4.T4.5.3.3.1 .ltx_text style="font-size:70%;"}     [1920]{#S4.T4.5.3.1.1 .ltx_text style="font-size:70%;"}$\times$[1440]{#S4.T4.5.3.1.2 .ltx_text style="font-size:70%;"}       [19]{#S4.T4.5.3.4.1 .ltx_text style="font-size:70%;"}           [21]{#S4.T4.5.3.5.1 .ltx_text style="font-size:70%;"}                                                                      [81]{#S4.T4.5.3.6.1 .ltx_text style="font-size:70%;"}                                                                      [81]{#S4.T4.5.3.7.1 .ltx_text style="font-size:70%;"}
  [iphone-lego2]{#S4.T4.6.4.2.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}     [23]{#S4.T4.6.4.3.1 .ltx_text style="font-size:70%;"}     [1920]{#S4.T4.6.4.1.1 .ltx_text style="font-size:70%;"}$\times$[1440]{#S4.T4.6.4.1.2 .ltx_text style="font-size:70%;"}       [19]{#S4.T4.6.4.4.1 .ltx_text style="font-size:70%;"}           [22]{#S4.T4.6.4.5.1 .ltx_text style="font-size:70%;"}                                                                      [78]{#S4.T4.6.4.6.1 .ltx_text style="font-size:70%;"}                                                                      [78]{#S4.T4.6.4.7.1 .ltx_text style="font-size:70%;"}
  [iphone-lego3]{#S4.T4.7.5.2.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}     [21]{#S4.T4.7.5.3.1 .ltx_text style="font-size:70%;"}     [1920]{#S4.T4.7.5.1.1 .ltx_text style="font-size:70%;"}$\times$[1440]{#S4.T4.7.5.1.2 .ltx_text style="font-size:70%;"}       [22]{#S4.T4.7.5.4.1 .ltx_text style="font-size:70%;"}           [28]{#S4.T4.7.5.5.1 .ltx_text style="font-size:70%;"}                                                                      [114]{#S4.T4.7.5.6.1 .ltx_text style="font-size:70%;"}                                                                     [114]{#S4.T4.7.5.7.1 .ltx_text style="font-size:70%;"}
  [iphone-pots1]{#S4.T4.8.6.2.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}     [25]{#S4.T4.8.6.3.1 .ltx_text style="font-size:70%;"}     [1920]{#S4.T4.8.6.1.1 .ltx_text style="font-size:70%;"}$\times$[1440]{#S4.T4.8.6.1.2 .ltx_text style="font-size:70%;"}       [19]{#S4.T4.8.6.4.1 .ltx_text style="font-size:70%;"}           [22]{#S4.T4.8.6.5.1 .ltx_text style="font-size:70%;"}                                                                      [76]{#S4.T4.8.6.6.1 .ltx_text style="font-size:70%;"}                                                                      [76]{#S4.T4.8.6.7.1 .ltx_text style="font-size:70%;"}
  [iphone-pots2]{#S4.T4.9.7.2.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}     [26]{#S4.T4.9.7.3.1 .ltx_text style="font-size:70%;"}     [1920]{#S4.T4.9.7.1.1 .ltx_text style="font-size:70%;"}$\times$[1440]{#S4.T4.9.7.1.2 .ltx_text style="font-size:70%;"}       [20]{#S4.T4.9.7.4.1 .ltx_text style="font-size:70%;"}           [23]{#S4.T4.9.7.5.1 .ltx_text style="font-size:70%;"}                                                                      [83]{#S4.T4.9.7.6.1 .ltx_text style="font-size:70%;"}                                                                      [83]{#S4.T4.9.7.7.1 .ltx_text style="font-size:70%;"}
  [pixel5-lamp]{#S4.T4.10.8.2.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}     [35]{#S4.T4.10.8.3.1 .ltx_text style="font-size:70%;"}    [1600]{#S4.T4.10.8.1.1 .ltx_text style="font-size:70%;"}$\times$[1200]{#S4.T4.10.8.1.2 .ltx_text style="font-size:70%;"}     [16]{#S4.T4.10.8.4.1 .ltx_text style="font-size:70%;"}          [23]{#S4.T4.10.8.5.1 .ltx_text style="font-size:70%;"}                                                                     [62]{#S4.T4.10.8.6.1 .ltx_text style="font-size:70%;"}                                                                     [67]{#S4.T4.10.8.7.1 .ltx_text style="font-size:70%;"}
  [pixel5-plant]{#S4.T4.11.9.2.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}    [30]{#S4.T4.11.9.3.1 .ltx_text style="font-size:70%;"}    [1600]{#S4.T4.11.9.1.1 .ltx_text style="font-size:70%;"}$\times$[1200]{#S4.T4.11.9.1.2 .ltx_text style="font-size:70%;"}     [15]{#S4.T4.11.9.4.1 .ltx_text style="font-size:70%;"}          [19]{#S4.T4.11.9.5.1 .ltx_text style="font-size:70%;"}                                                                     [52]{#S4.T4.11.9.6.1 .ltx_text style="font-size:70%;"}                                                                     [58]{#S4.T4.11.9.7.1 .ltx_text style="font-size:70%;"}
  [pixel5-table]{#S4.T4.12.10.2.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}   [23]{#S4.T4.12.10.3.1 .ltx_text style="font-size:70%;"}   [1600]{#S4.T4.12.10.1.1 .ltx_text style="font-size:70%;"}$\times$[1200]{#S4.T4.12.10.1.2 .ltx_text style="font-size:70%;"}   [16]{#S4.T4.12.10.4.1 .ltx_text style="font-size:70%;"}         [20]{#S4.T4.12.10.5.1 .ltx_text style="font-size:70%;"}                                                                    [55]{#S4.T4.12.10.6.1 .ltx_text style="font-size:70%;"}                                                                    [62]{#S4.T4.12.10.7.1 .ltx_text style="font-size:70%;"}
  [s20-bike]{#S4.T4.13.11.2.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}       [43]{#S4.T4.13.11.3.1 .ltx_text style="font-size:70%;"}   [1920]{#S4.T4.13.11.1.1 .ltx_text style="font-size:70%;"}$\times$[1440]{#S4.T4.13.11.1.2 .ltx_text style="font-size:70%;"}   [21]{#S4.T4.13.11.4.1 .ltx_text style="font-size:70%;"}         [26]{#S4.T4.13.11.5.1 .ltx_text style="font-size:70%;"}                                                                    [79]{#S4.T4.13.11.6.1 .ltx_text style="font-size:70%;"}                                                                    [90]{#S4.T4.13.11.7.1 .ltx_text style="font-size:70%;"}
  [s20-bikerack]{#S4.T4.14.12.2.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}   [32]{#S4.T4.14.12.3.1 .ltx_text style="font-size:70%;"}   [1920]{#S4.T4.14.12.1.1 .ltx_text style="font-size:70%;"}$\times$[1440]{#S4.T4.14.12.1.2 .ltx_text style="font-size:70%;"}   [20]{#S4.T4.14.12.4.1 .ltx_text style="font-size:70%;"}         [26]{#S4.T4.14.12.5.1 .ltx_text style="font-size:70%;"}                                                                    [71]{#S4.T4.14.12.6.1 .ltx_text style="font-size:70%;"}                                                                    [79]{#S4.T4.14.12.7.1 .ltx_text style="font-size:70%;"}
  [s20-sign]{#S4.T4.15.13.2.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}       [28]{#S4.T4.15.13.3.1 .ltx_text style="font-size:70%;"}   [1920]{#S4.T4.15.13.1.1 .ltx_text style="font-size:70%;"}$\times$[1440]{#S4.T4.15.13.1.2 .ltx_text style="font-size:70%;"}   [21]{#S4.T4.15.13.4.1 .ltx_text style="font-size:70%;"}         [36]{#S4.T4.15.13.5.1 .ltx_text style="font-size:70%;"}                                                                    [162]{#S4.T4.15.13.6.1 .ltx_text style="font-size:70%;"}                                                                   [210]{#S4.T4.15.13.7.1 .ltx_text style="font-size:70%;"}
  [average]{#S4.T4.15.14.1.1.1 .ltx_text .ltx_font_smallcaps style="font-size:70%;"}                                                                                                                                                                                             [19]{#S4.T4.15.14.1.4.1 .ltx_text style="font-size:70%;"}       [24]{#S4.T4.15.14.1.5.1 .ltx_text style="font-size:70%;"}                                                                  [83]{#S4.T4.15.14.1.6.1 .ltx_text style="font-size:70%;"}                                                                  [91]{#S4.T4.15.14.1.7.1 .ltx_text style="font-size:70%;"}

Report issue for preceding element
:::

::: {#S4.SS2.SSS0.Px8 .section .ltx_paragraph}
##### Timing tests {#timing-tests .ltx_title .ltx_title_paragraph}

Report issue for preceding element

::: {#S4.SS2.SSS0.Px8.p1 .ltx_para}
The training times corresponding to selected experiments
in [[Table]{.ltx_text .ltx_ref_tag} [3]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S4.T3 "In Re-rendered data ‣ 4.1 Synthetic Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
(which were computed on NVidia A100 GPUs) are shown
in [[Table]{.ltx_text .ltx_ref_tag} [4]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S4.T4 "In Results ‣ 4.2 Smartphone Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}.
The increase in training times (here shown for a single test run) varies
significantly by case. On the average, our full method with
$N_{blur} = 5$ samples is approximately five times slower than the
baseline. Disabling rolling shutter compensation only has a small, and
possibly random effect. However, disabling the motion blur compensation
shows that our rolling shutter has a low overhead (26%) compared to the
baseline. We also note that our CUDA implementation was not yet profiled
for new bottlenecks, and could have significant potential for further
speed improvement for motion blur compensation. The memory consumption
of our approach is not significantly higher than for the baseline
Splatfacto model. For development purposes, we also successfully trained
on consumer grade GPUs, such as NVidia RTX 4060 TI with 16 GB of VRAM.

Report issue for preceding element
:::
:::
:::

::: {#S5 .section .ltx_section}
[5 ]{.ltx_tag .ltx_tag_section}Discussion and Conclusion {#discussion-and-conclusion .ltx_title .ltx_title_section}
--------------------------------------------------------

Report issue for preceding element

::: {#S5.p1 .ltx_para}
We demonstrated how blurring and rolling shutter effects can be
efficiently implemented in the Gaussian Splatting (3DGS) framework. We
quantitatively and qualitatively demonstrated that the method improves
over the 3DGS Splatfacto baseline in both synthetic and real data
experiments. Furthermore, we showed that our method outperforms the
previous NeRF-based state-of-the-art,
BAD-NeRF \[[40](https://arxiv.org/html/2403.13327v3#bib.bib40){.ltx_ref}\]
on synthetic data. We also demonstrated superior performance and
robustness compared to baselines where the data was first deblurred or
de-rolled using modern deep-learning-based methods,
MPR \[[50](https://arxiv.org/html/2403.13327v3#bib.bib50){.ltx_ref}\] or
CVR \[[6](https://arxiv.org/html/2403.13327v3#bib.bib6){.ltx_ref}\].

Report issue for preceding element
:::

::: {#S5.p2 .ltx_para}
Incorporating learning-based aspects directly into the 3D model
generation instead of the 2D input data is a new and promising approach
studied in, *e.g*.[]{#S5.p2.1.2 .ltx_text},
\[[41](https://arxiv.org/html/2403.13327v3#bib.bib41){.ltx_ref},
[42](https://arxiv.org/html/2403.13327v3#bib.bib42){.ltx_ref}\]. We
believe that, in the context of differentiable rendering and 3D
reconstruction, this approach is likely to prove superior to
learning-based image manipulation as a pre-processing step. Modelling
local linear trajectories with more complex spline-based shapes as in
\[[40](https://arxiv.org/html/2403.13327v3#bib.bib40){.ltx_ref},
[18](https://arxiv.org/html/2403.13327v3#bib.bib18){.ltx_ref}\] could be
an additional follow up work.

Report issue for preceding element
:::

::: {#S5.p3 .ltx_para}
Data capture in-the-wild is typically done on smart phones with a
rolling shutter sensor and during relative motion. These effects are
rarely represented well in modern benchmark data for inverse rendering,
limiting the real-world use of 3DGS based methods. Our work represents a
significant step forward in the integration of motion blur and rolling
shutter corrections within the 3DGS framework, opening up new avenues
for research and application in differentiable rendering and 3D scene
reconstruction.

Report issue for preceding element
:::

::: {#S5.p4 .ltx_para}
The source code of our implementation is available at
<https://github.com/SpectacularAI/3dgs-deblur>. Additionally, the
smartphone data set can be accessed at
<https://doi.org/10.5281/zenodo.10848124> and our Deblur-NeRF data set
variant at <https://doi.org/10.5281/zenodo.10847884>.

Report issue for preceding element
:::
:::

::: {#Sx1 .section .ltx_section}
Acknowledgements {#acknowledgements .ltx_title .ltx_title_section}
----------------

Report issue for preceding element

::: {#Sx1.p1 .ltx_para}
MT was supported by the Research Council of Finland Flagship programme:
Finnish Center for Artificial Intelligence (FCAI). AS acknowledges
funding from the Research Council of Finland (grant id 339730). We
acknowledge CSC -- IT Center for Science, Finland, for computational
resources.

Report issue for preceding element
:::

::: {.ltx_pagination .ltx_role_newpage}
:::
:::

::: {#bib .section .ltx_bibliography}
References {#references .ltx_title .ltx_title_bibliography}
----------

Report issue for preceding element

-   [\[1\]]{#bib.bib1}
    ↑
    [ Spectacular AI mapping tools (2024),
    <https://spectacularai.github.io/docs/sdk/tools/nerf.html>,
    accessed: 2024-03-01 ]{.ltx_bibblock}
-   [\[2\]]{#bib.bib2}
    ↑
    [ Cai, J.F., Ji, H., Liu, C., Shen, Z.: Blind motion deblurring from
    a single image using sparse approximation. In: IEEE Conference on
    Computer Vision and Pattern Recognition (CVPR). pp. 104--111. IEEE
    (2009) ]{.ltx_bibblock}
-   [\[3\]]{#bib.bib3}
    ↑
    [ Chakrabarti, A.: A neural approach to blind motion deblurring. In:
    European Conference on Computer Vision (ECCV). pp. 221--235 (2016)
    ]{.ltx_bibblock}
-   [\[4\]]{#bib.bib4}
    ↑
    [ Community, B.O.: Blender -- a 3D modelling and rendering package.
    Blender Foundation (2018),
    [http://www.blender.org](http://www.blender.org/){.ltx_ref .ltx_url
    .ltx_font_typewriter} ]{.ltx_bibblock}
-   [\[5\]]{#bib.bib5}
    ↑
    [ Dai, P., Zhang, Y., Yu, X., Lyu, X., Qi, X.: Hybrid neural
    rendering for large-scale scenes with motion blur. In: IEEE/CVF
    Conference on Computer Vision and Pattern Recognition (CVPR). pp.
    154--164 (2023) ]{.ltx_bibblock}
-   [\[6\]]{#bib.bib6}
    ↑
    [ Fan, B., Dai, Y., Zhang, Z., Liu, Q., He, M.: Context-aware video
    reconstruction for rolling shutter cameras. In: IEEE/CVF Conference
    on Computer Vision and Pattern Recognition (CVPR). pp. 17551--17561
    (2022) ]{.ltx_bibblock}
-   [\[7\]]{#bib.bib7}
    ↑
    [ Fu, Y., Liu, S., Kulkarni, A., Kautz, J., Efros, A.A., Wang, X.:
    COLMAP-Free 3D Gaussian splatting. arXiv preprint arXiv:2312.07504
    (2023) ]{.ltx_bibblock}
-   [\[8\]]{#bib.bib8}
    ↑
    [ Gong, D., Yang, J., Liu, L., Zhang, Y., Reid, I., Shen, C.,
    van den Hengel, A., Shi, Q.: From motion blur to motion flow: A deep
    learning solution for removing heterogeneous motion blur. In: IEEE
    Conference on Computer Vision and Pattern Recognition (CVPR). pp.
    2319--2328 (2017) ]{.ltx_bibblock}
-   [\[9\]]{#bib.bib9}
    ↑
    [ Grundmann, M., Kwatra, V., Castro, D., Essa, I.: Calibration-free
    rolling shutter removal. In: IEEE International Conference on
    Computational Photography (ICCP). pp. 1--8 (2012) ]{.ltx_bibblock}
-   [\[10\]]{#bib.bib10}
    ↑
    [ Hedborg, J., Forsén, P.E., Felsberg, M., Ringaby, E.: Rolling
    shutter bundle adjustment. In: IEEE Conference on Computer Vision
    and Pattern Recognition (CVPR). pp. 1434--1441 (2012)
    ]{.ltx_bibblock}
-   [\[11\]]{#bib.bib11}
    ↑
    [ Kaipio, J., Somersalo, E.: Statistical and computational inverse
    problems. Springer (2004) ]{.ltx_bibblock}
-   [\[12\]]{#bib.bib12}
    ↑
    [ Keetha, N., Karhade, J., Jatavallabhula, K.M., Yang, G., Scherer,
    S., Ramanan, D., Luiten, J.: SplaTAM: Splat, track & map 3D
    Gaussians for dense RGB-D SLAM. In: IEEE/CVF Conference on Computer
    Vision and Pattern Recognition (CVPR). pp. 21357--21366 (2024)
    ]{.ltx_bibblock}
-   [\[13\]]{#bib.bib13}
    ↑
    [ Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3D Gaussian
    Splatting for real-time radiance field rendering. ACM Transactions
    on Graphics (TOG) [42]{#bib.bib13.1.1 .ltx_text .ltx_font_bold}(4)
    (2023) ]{.ltx_bibblock}
-   [\[14\]]{#bib.bib14}
    ↑
    [ Kim, H., Song, M., Lee, D., Kim, P.: Visual-inertial odometry
    priors for bundle-adjusting neural radiance fields. In:
    International Conference on Control, Automation and Systems (ICCAS).
    pp. 1131--1136 (2022) ]{.ltx_bibblock}
-   [\[15\]]{#bib.bib15}
    ↑
    [ Kupyn, O., Budzan, V., Mykhailych, M., Mishkin, D., Matas, J.:
    DeblurGAN: Blind motion deblurring using conditional adversarial
    networks. In: IEEE Conference on Computer Vision and Pattern
    Recognition (CVPR). pp. 8183--8192 (2018) ]{.ltx_bibblock}
-   [\[16\]]{#bib.bib16}
    ↑
    [ Lao, Y., Ait-Aider, O.: A robust method for strong rolling shutter
    effects correction using lines with automatic feature selection. In:
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
    pp. 4795--4803 (2018) ]{.ltx_bibblock}
-   [\[17\]]{#bib.bib17}
    ↑
    [ Lee, B., Lee, H., Sun, X., Ali, U., Park, E.: Deblurring 3D
    Gaussian Splatting. arXiv preprint arXiv:2401.00834 (2024)
    ]{.ltx_bibblock}
-   [\[18\]]{#bib.bib18}
    ↑
    [ Li, M., Wang, P., Zhao, L., Liao, B., Liu, P.: USB-neRF: Unrolling
    shutter bundle adjusted neural radiance fields. In: International
    Conference on Learning Representations (ICLR) (2024)
    ]{.ltx_bibblock}
-   [\[19\]]{#bib.bib19}
    ↑
    [ Liang, C.K., Chang, L.W., Chen, H.H.: Analysis and compensation of
    rolling shutter effect. IEEE Transactions on Image Processing
    [17]{#bib.bib19.1.1 .ltx_text .ltx_font_bold}(8), 1323--1330 (2008)
    ]{.ltx_bibblock}
-   [\[20\]]{#bib.bib20}
    ↑
    [ Liao, B., Qu, D., Xue, Y., Zhang, H., Lao, Y.: Revisiting rolling
    shutter bundle adjustment: Toward accurate and fast solution. In:
    IEEE/CVF Conference on Computer Vision and Pattern Recognition
    (CVPR). pp. 4863--4871 (2023) ]{.ltx_bibblock}
-   [\[21\]]{#bib.bib21}
    ↑
    [ Lin, C.H., Ma, W.C., Torralba, A., Lucey, S.: BARF:
    Bundle-adjusting neural radiance fields. IEEE/CVF International
    Conference on Computer Vision (ICCV) pp. 5721--5731 (2021)
    ]{.ltx_bibblock}
-   [\[22\]]{#bib.bib22}
    ↑
    [ Liu, P., Cui, Z., Larsson, V., Pollefeys, M.: Deep shutter
    unrolling network. In: IEEE/CVF Conference on Computer Vision and
    Pattern Recognition (CVPR). pp. 5940--5948 (2020) ]{.ltx_bibblock}
-   [\[23\]]{#bib.bib23}
    ↑
    [ Lovegrove, S., Patron-Perez, A., Sibley, G.: Spline fusion: A
    continuous-time representation for visual-inertial fusion with
    application to rolling shutter cameras. In: Proceedings of the
    British Machine Vision Conference (BMVC). pp. 93.1--93.11 (2013)
    ]{.ltx_bibblock}
-   [\[24\]]{#bib.bib24}
    ↑
    [ Lowe, D.G.: Distinctive image features from scale-invariant
    keypoints. International Journal of Computer Vision
    [60]{#bib.bib24.1.1 .ltx_text .ltx_font_bold}(2), 91--110 (2004)
    ]{.ltx_bibblock}
-   [\[25\]]{#bib.bib25}
    ↑
    [ Ma, L., Li, X., Liao, J., Zhang, Q., Wang, X., Wang, J., Sander,
    P.V.: Deblur-NeRF: Neural radiance fields from blurry images. In:
    IEEE/CVF Conference on Computer Vision and Pattern Recognition
    (CVPR). pp. 12861--12870 (2022) ]{.ltx_bibblock}
-   [\[26\]]{#bib.bib26}
    ↑
    [ Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T.,
    Ramamoorthi, R., Ng, R.: NeRF: Representing scenes as neural
    radiance fields for view synthesis. In: European Conference on
    Computer Vision (ECCV). pp. 405--421 (2020) ]{.ltx_bibblock}
-   [\[27\]]{#bib.bib27}
    ↑
    [ Mohan M.R., M., Rajagopalan, A., Seetharaman, G.: Going
    unconstrained with rolling shutter deblurring. In: IEEE
    International Conference on Computer Vision (ICCV). pp. 4030--4038
    (2017) ]{.ltx_bibblock}
-   [\[28\]]{#bib.bib28}
    ↑
    [ Park, K., Henzler, P., Mildenhall, B., Barron, J.T.,
    Martin-Brualla, R.: CamP: Camera preconditioning for neural radiance
    fields. ACM Transactions on Graphics (TOG) [42]{#bib.bib28.1.1
    .ltx_text .ltx_font_bold}(6), 1--11 (2023) ]{.ltx_bibblock}
-   [\[29\]]{#bib.bib29}
    ↑
    [ Rengarajan, V., Rajagopalan, A.N., Aravind, R.: From bows to
    arrows: Rolling shutter rectification of urban scenes. In: IEEE
    Conference on Computer Vision and Pattern Recognition (CVPR). pp.
    2773--2781 (2016) ]{.ltx_bibblock}
-   [\[30\]]{#bib.bib30}
    ↑
    [ Schölkopf, B., Platt, J., Hofmann, T.: Blind motion deblurring
    using image statistics. In: Advances in Neural Information
    Processing Systems 19. pp. 841--848 (2007) ]{.ltx_bibblock}
-   [\[31\]]{#bib.bib31}
    ↑
    [ Schönberger, J.L., Frahm, J.M.: Structure-from-motion revisited.
    In: IEEE Conference on Computer Vision and Pattern Recognition
    (CVPR) (2016) ]{.ltx_bibblock}
-   [\[32\]]{#bib.bib32}
    ↑
    [ Schubert, D., Demmel, N., von Stumberg, L., Usenko, V., Cremers,
    D.: Rolling-shutter modelling for visual-inertial odometry. In:
    International Conference on Intelligent Robots and Systems (IROS)
    (2019) ]{.ltx_bibblock}
-   [\[33\]]{#bib.bib33}
    ↑
    [ Seiskari, O., Rantalankila, P., Kannala, J., Ylilammi, J., Rahtu,
    E., Solin, A.: HybVIO: Pushing the limits of real-time
    visual-inertial odometry. In: IEEE/CVF Winter Conference on
    Applications of Computer Vision (WACV). pp. 287--296. IEEE Winter
    Conference on Applications of Computer Vision, IEEE (2022)
    ]{.ltx_bibblock}
-   [\[34\]]{#bib.bib34}
    ↑
    [ Shan, Q., Jia, J., Agarwala, A.: High-quality motion deblurring
    from a single image. ACM Transactions on Graphics (TOG)
    [27]{#bib.bib34.1.1 .ltx_text .ltx_font_bold}(3), 1--10 (2008)
    ]{.ltx_bibblock}
-   [\[35\]]{#bib.bib35}
    ↑
    [ Su, S., Heidrich, W.: Rolling shutter motion deblurring. In: IEEE
    Conference on Computer Vision and Pattern Recognition (CVPR). pp.
    1529--1537 (2015) ]{.ltx_bibblock}
-   [\[36\]]{#bib.bib36}
    ↑
    [ Tai, Y.W., Tan, P., Brown, M.S.: Richardson--Lucy deblurring for
    scenes under a projective motion path. IEEE Transactions on Pattern
    Analysis and Machine Intelligence [33]{#bib.bib36.1.1 .ltx_text
    .ltx_font_bold}(8), 1603--1618 (2011) ]{.ltx_bibblock}
-   [\[37\]]{#bib.bib37}
    ↑
    [ Tancik, M., Weber, E., Ng, E., Li, R., Yi, B., Kerr, J., Wang, T.,
    Kristoffersen, A., Austin, J., Salahi, K., Ahuja, A., McAllister,
    D., Kanazawa, A.: Nerfstudio: A modular framework for neural
    radiance field development. In: ACM SIGGRAPH 2023 Conference
    Proceedings (2023) ]{.ltx_bibblock}
-   [\[38\]]{#bib.bib38}
    ↑
    [ Triggs, B., McLauchlan, P.F., Hartley, R.I., Fitzgibbon, A.W.:
    Bundle adjustment - a modern synthesis. In: Proceedings of the
    International Workshop on Vision Algorithms: Theory and Practice.
    pp. 298--372. ICCV '99, Springer-Verlag (2000) ]{.ltx_bibblock}
-   [\[39\]]{#bib.bib39}
    ↑
    [ Vasu, S., Mohan M.R., M., Rajagopalan, A.: Occlusion-aware rolling
    shutter rectification of 3D scenes. In: IEEE/CVF Conference on
    Computer Vision and Pattern Recognition (CVPR). pp. 636--645 (2018)
    ]{.ltx_bibblock}
-   [\[40\]]{#bib.bib40}
    ↑
    [ Wang, P., Zhao, L., Ma, R., Liu, P.: BAD-NeRF: Bundle adjusted
    deblur neural radiance fields. In: IEEE/CVF Conference on Computer
    Vision and Pattern Recognition (CVPR). pp. 4170--4179 (2023)
    ]{.ltx_bibblock}
-   [\[41\]]{#bib.bib41}
    ↑
    [ Weber, E., Holynski, A., Jampani, V., Saxena, S., Snavely, N.,
    Kar, A., Kanazawa, A.: NeRFiller: Completing scenes via generative
    3D inpainting. In: IEEE/CVF Conference on Computer Vision and
    Pattern Recognition (CVPR). pp. 20731--20741 (2024) ]{.ltx_bibblock}
-   [\[42\]]{#bib.bib42}
    ↑
    [ Wu, R., Mildenhall, B., Henzler, P., Park, K., Gao, R., Watson,
    D., Srinivasan, P.P., Verbin, D., Barron, J.T., Poole, B., Holynski,
    A.: ReconFusion: 3D reconstruction with diffusion priors. arXiv
    preprint arXiv:2312.02981 (2023) ]{.ltx_bibblock}
-   [\[43\]]{#bib.bib43}
    ↑
    [ Xie, T., Zong, Z., Qiu, Y., Li, X., Feng, Y., Yang, Y., Jiang, C.:
    PhysGaussian: Physics-integrated 3D Gaussians for generative
    dynamics. arXiv preprint arXiv:2311.12198 (2023) ]{.ltx_bibblock}
-   [\[44\]]{#bib.bib44}
    ↑
    [ Xu, L., Jia, J.: Two-phase kernel estimation for robust motion
    deblurring. In: European Conference on Computer Vision (ECCV). pp.
    157--170 (2010) ]{.ltx_bibblock}
-   [\[45\]]{#bib.bib45}
    ↑
    [ Yan, C., Qu, D., Xu, D., Zhao, B., Wang, Z., Wang, D., Li, X.:
    GS-SLAM: Dense visual SLAM with 3D Gaussian splatting. In: IEEE/CVF
    Conference on Computer Vision and Pattern Recognition (CVPR). pp.
    19595--19604 (2024) ]{.ltx_bibblock}
-   [\[46\]]{#bib.bib46}
    ↑
    [ Ye, V., Kanazawa, A.: Mathematical supplement for the
    [gsplat]{#bib.bib46.2.1 .ltx_text .ltx_markedasmath
    .ltx_font_typewriter} library (2023) ]{.ltx_bibblock}
-   [\[47\]]{#bib.bib47}
    ↑
    [ Ye, V., Turkulainen, M., the Nerfstudio team: gsplat,
    <https://github.com/nerfstudio-project/gsplat> ]{.ltx_bibblock}
-   [\[48\]]{#bib.bib48}
    ↑
    [ Ye, Z., Li, W., Liu, S., Qiao, P., Dou, Y.: AbsGS: Recovering fine
    details for 3D Gaussian splatting. arXiv preprint arXiv:2404.10484
    (2024) ]{.ltx_bibblock}
-   [\[49\]]{#bib.bib49}
    ↑
    [ Yu, Z., Chen, A., Huang, B., Sattler, T., Geiger, A.:
    Mip-Splatting: Alias-free 3D Gaussian splatting. In: IEEE/CVF
    Conference on Computer Vision and Pattern Recognition (CVPR). pp.
    19447--19456 (2024) ]{.ltx_bibblock}
-   [\[50\]]{#bib.bib50}
    ↑
    [ Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang,
    M.H., Shao, L.: Multi-stage progressive image restoration. In:
    IEEE/CVF Conference on Computer Vision and Pattern Recognition
    (CVPR). pp. 14816--14826 (2021) ]{.ltx_bibblock}
:::

::: {.ltx_pagination .ltx_role_newpage}
:::

::: {#p2 .ltx_para .ltx_align_center}
[Supplementary Material[\
]{#p2.1.1.1 .ltx_text .ltx_font_medium}]{#p2.1.1 .ltx_text
.ltx_font_bold style="font-size:144%;"}

Report issue for preceding element
:::

::: {#Pt0.A1 .section .ltx_appendix}
[Appendix 0.A ]{.ltx_tag .ltx_tag_appendix}Method Details {#appendix-0.a-method-details .ltx_title .ltx_title_appendix}
---------------------------------------------------------

Report issue for preceding element

::: {#Pt0.A1.SS1 .section .ltx_subsection}
### [0.A.1 ]{.ltx_tag .ltx_tag_subsection}Gaussian parametrization {#a.1-gaussian-parametrization .ltx_title .ltx_title_subsection}

Report issue for preceding element

::: {#Pt0.A1.SS1.p1 .ltx_para}
In [gsplat]{#Pt0.A1.SS1.p1.1.1 .ltx_text .ltx_font_typewriter}, the
Gaussian covariances $\Sigma$ in world coordinates are parametrized as

Report issue for preceding element

  -- ------------------------------------------------------------------------------ -- -----------------------------------------------------
     $$\Sigma = {R{(q)}{diag}{({{sigmoid}{(s_{1},s_{2},s_{3})}})}R{(q)}^{\top}}$$      [(11)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- ------------------------------------------------------------------------------ -- -----------------------------------------------------

where $q \in {\mathbb{R}}^{4}$ is a normalized quaternion corresponding
to a rotation matrix ${R{(q)}} \in {{SO}{(3)}}$ and
${(s_{1},s_{2},s_{3})} \in {\mathbb{R}}$ are scale parameters. This is
not changed in our implementation, but we use $\Sigma$ instead of the
above parameters in this paper for brevity and clarity.

Report issue for preceding element
:::
:::

::: {#Pt0.A1.SS2 .section .ltx_subsection}
### [0.A.2 ]{.ltx_tag .ltx_tag_subsection}Transforming Gaussians from world to pixel coordinates {#a.2-transforming-gaussians-from-world-to-pixel-coordinates .ltx_title .ltx_title_subsection}

Report issue for preceding element

::: {#Pt0.A1.SS2.p1 .ltx_para}
In 3DGS, the rendering equation can be written as,

Report issue for preceding element

  -- ----------------------------------------------------------------------------------- -- -----------------------------------------------------
     $${C_{i}{(x,y,P,\mathcal{G})}} = {r{(x,y,{\pi{({\hat{p}{(\mathcal{G},P)}})}})}}$$      [(12)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- ----------------------------------------------------------------------------------- -- -----------------------------------------------------

where
$\hat{p}:{{(\mu,\Sigma,\theta,P)}\mapsto{(\hat{\mu},\hat{\Sigma},c)}}$
maps each (visible) Gaussian from the world coordinate system to the
*camera coordinate* system

Report issue for preceding element

  -- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----------------------------------------------------
     $${{\hat{\mu}}_{i,j} = {R_{i}^{\top}{({\mu_{j} - p_{i}})}}},{{{\hat{\Sigma}}_{i,j} = {R_{i}^{\top}\Sigma_{j}R_{i}}},{c_{i,j} = {\theta_{j}{(\frac{\mu_{j} - p_{i}}{\|{\mu_{j} - p_{i}}\|})}}}}$$      [(13)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----------------------------------------------------

where the colour $c$ is computed by evaluation the spherical harmonic
function at the normalized viewing direction. We use spherical harmonics
of degree three. The function
$\pi:{{(\hat{\mu},\hat{\Sigma})}\mapsto{(\mu^{\prime},d,\Sigma^{\prime})}}$
projects Gaussians from camera coordinates to pixel coordinates with
depth:

Report issue for preceding element

  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----------------------------------------------------
     $${{d = {\overset{\sim}{\mu}}_{z}},{{\mu^{\prime} = {({{\overset{\sim}{\mu}}_{x}/d},{{\overset{\sim}{\mu}}_{y}/d})}},{\Sigma^{\prime} = {J_{i}\hat{\Sigma}J_{i}^{\top}}}}},$$      [(14)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----------------------------------------------------

where $\overset{\sim}{\mu} = {K_{i}\hat{\mu}}$ and
$J_{i} = {J^{\prime}K_{i}}$ is the Jacobian matrix of the pinhole camera
projection $\hat{\mu}\mapsto\mu^{\prime}$ with intrisic camera matrix
$K_{i}$:

Report issue for preceding element

  -- ------------------------------------------------ -- -----------------------------------------------------
     $${{J^{\prime} = {\frac{1}{d}\begin{bmatrix}        [(15)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
     1 & \;0 & {- {{\overset{\sim}{\mu}}_{x}/d}} \\      
     0 & \;1 & {- {{\overset{\sim}{\mu}}_{y}/d}} \\      
     \end{bmatrix}}},{K_{i} = \begin{bmatrix}            
     f_{x} & 0 & c_{x} \\                                
     0 & f_{y} & c_{y} \\                                
     0 & 0 & 1 \\                                        
     \end{bmatrix}}}.$$                                  
  -- ------------------------------------------------ -- -----------------------------------------------------

Note that unlike the original [gsplat]{#Pt0.A1.SS2.p1.9.1 .ltx_text
.ltx_font_typewriter}
\[[46](https://arxiv.org/html/2403.13327v3#bib.bib46){.ltx_ref}\] and
the Inria implementations, we do not use the OpenGL NDC coordinate
system as an intermediate step between projecting Gaussians to pixel
coordinates.

Report issue for preceding element
:::

::: {#Pt0.A1.SS2.p2 .ltx_para}
The Gaussian with low depth $d < d_{\min}$ or pixel coordinates
$(\mu_{x}^{\prime},\mu_{y}^{\prime})$ too far outside the image
boundaries ${\lbrack 0,W)} \times {\lbrack 0,H)}$ are discarded in the
next rendering phases represented by the function $r$.

Report issue for preceding element
:::
:::

::: {#Pt0.A1.SS3 .section .ltx_subsection}
### [0.A.3 ]{.ltx_tag .ltx_tag_subsection}Differentiation with respect to the camera pose {#a.3-differentiation-with-respect-to-the-camera-pose .ltx_title .ltx_title_subsection}

Report issue for preceding element

::: {#Pt0.A1.SS3.p1 .ltx_para}
We seek to differentiate the rendering equation [[Eq.]{.ltx_text
.ltx_ref_tag} [12]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A1.E12 "In 0.A.2 Transforming Gaussians from world to pixel coordinates ‣ Appendix 0.A Method Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
with respect to the current camera pose parameters
$P_{i} \in {{SE}{(3)}}$. The key to camera pose optimization is
differentiating the intermediate projection terms [[Eq.]{.ltx_text
.ltx_ref_tag} [13]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A1.E13 "In 0.A.2 Transforming Gaussians from world to pixel coordinates ‣ Appendix 0.A Method Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
with respect to some parametrization of
$P_{i} = {\lbrack\left. R_{i} \middle| p_{i} \right.\rbrack}$. First,
note that the Jacobian[^1^[[^1^[1]{.ltx_tag .ltx_tag_note}Slight abuse
of notation: we use $\frac{\partial}{\partial\nu}$ notation for all
derivatives, including Jacobian matrices and multi-dimensional tensors,
such as
$\frac{\partial{\hat{\Sigma}}_{i,j}}{\partial p_{i}}$]{.ltx_note_content}]{.ltx_note_outer}]{#footnote1
.ltx_note .ltx_role_footnote} of the Gaussian mean in camera coordinates
with respect to the camera center $p_{i}$ is

Report issue for preceding element

  -- --------------------------------------------------------------------------------------------------------------------------------- -- -----------------------------------------------------
     $$\frac{\partial{\hat{\mu}}_{i,j}}{\partial p_{i}} = {- R_{i}^{\top}} = {- \frac{\partial{\hat{\mu}}_{i,j}}{\partial\mu_{j}}}$$      [(16)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- --------------------------------------------------------------------------------------------------------------------------------- -- -----------------------------------------------------

and if we neglect the effect of $p_{i}$ on $c_{i,j}$,
*i.e*.[]{#Pt0.A1.SS3.p1.6.2 .ltx_text}, the view-dependency of the
colors, and on the covariances ${\hat{\Sigma}}_{i,j}$, we can
approximate:

Report issue for preceding element

  -- ----------------------------------------- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     $\frac{\partial C_{i}}{\partial p_{i}}$   $= {\sum\limits_{j}\left( {{\frac{\partial C_{i}}{\partial{\hat{\mu}}_{i,j}}\frac{\partial{\hat{\mu}}_{i,j}}{\partial p_{i}}} + {\frac{\partial C_{i}}{\partial{\hat{\Sigma}}_{i,j}}\frac{\partial{\hat{\Sigma}}_{i,j}}{\partial p_{i}}} + {\frac{\partial C_{i}}{\partial c_{i,j}}\frac{\partial c_{i,j}}{\partial p_{i}}}} \right)}$   
                                               ${\approx {\sum\limits_{j}{\frac{\partial C_{i}}{\partial{\hat{\mu}}_{i,j}}\frac{\partial{\hat{\mu}}_{i,j}}{\partial p_{i}}}} = {- {\sum\limits_{j}{\frac{\partial C_{i}}{\partial{\hat{\mu}}_{i,j}}\frac{\partial{\hat{\mu}}_{i,j}}{\mu_{j}}}}} \approx {- {\sum\limits_{j}\frac{\partial C_{i}}{\partial\mu_{j}}}}},$                  
  -- ----------------------------------------- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

that is, moving the camera to direction $\Delta p$ is approximately the
same as moving all the visible Gaussians to the opposite direction
$- {\Delta p}$.

Report issue for preceding element
:::

::: {#Pt0.A1.SS3.p2 .ltx_para}
Furthermore, using [[Eq.]{.ltx_text .ltx_ref_tag} [16]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A1.E16 "In 0.A.3 Differentiation with respect to the camera pose ‣ Appendix 0.A Method Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
and ignoring effects on color-dependency, we can use the following
expression

Report issue for preceding element

  -- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----------------------------------------------------
     $$\frac{\partial C_{i}}{\partial\mu_{j}} = {{\frac{\partial C_{i}}{\partial{\hat{\mu}}_{i,j}}\frac{\partial{\hat{\mu}}_{i,j}}{\partial\mu_{j}}} + {\frac{\partial C_{i}}{\partial c_{i,j}}\frac{\partial c_{i,j}}{\partial\mu_{j}}}} \approx {\frac{\partial C_{i}}{\partial{\hat{\mu}}_{i,j}}\frac{\partial{\hat{\mu}}_{i,j}}{\partial\mu_{j}}} = {\frac{\partial C_{i}}{\partial{\hat{\mu}}_{i,j}}R_{i}^{\top}}$$      [(17)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----------------------------------------------------

The derivative of the rendering equation with respect to the rotation of
the camera pose can also be expressed by neglecting the effect of small
rotations on the shape of ${\hat{\Sigma}}_{i,j}$ of the Gaussians in
camera coordinates. The derivative of the pixel colour $C_{i}$ with
respect to any rotation matrix $R_{i}$ component $\nu$ becomes

Report issue for preceding element
:::

::: {#Pt0.A1.SS3.p3 .ltx_para}
  -- -------------------------------------- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     $\frac{\partial C_{i}}{\partial\nu}$   $\approx {\sum\limits_{j}{\frac{\partial C_{i}}{\partial{\hat{\mu}}_{i,j}}\frac{\partial{\hat{\mu}}_{i,j}}{\partial\nu}}} \approx {\sum\limits_{j}{\frac{\partial C_{i}}{\partial\mu_{j}}R_{i}\frac{\partial{\hat{\mu}}_{i,j}}{\partial\nu}}} = {\sum\limits_{j}{\frac{\partial C_{i}}{\partial\mu_{j}}R_{i}\left( \frac{\partial R_{i}}{\partial\nu} \right)^{\top}{({\mu_{j} - p_{i}})}}}$   
                                            $= {\sum\limits_{j}{\frac{\partial C_{i}}{\partial\mu_{j}}R_{i}\left( \frac{\partial R_{i}}{\partial\nu} \right)^{\top}R_{i}{\hat{\mu}}_{i,j}}} = {- {\sum\limits_{j}{\frac{\partial C_{i}}{\partial\mu_{j}}\frac{\partial R_{i}}{\partial\nu}{\hat{\mu}}_{i,j}}}}$                                                                                                                            
  -- -------------------------------------- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

The benefit of this approach is that derivatives with respect to both
rotation $R_{i}$ and translation $p_{i}$ of the camera can be
approximated in terms of the derivatives
$\frac{\partial C_{i}}{\partial\mu_{j}}$ with respect to the Gaussian
means, and other properties that are readily available in the 3DGS
backwards pass.

Report issue for preceding element
:::

::: {#Pt0.A1.SS3.p4 .ltx_para}
The above formulas hold for camera-to-world transformations
$P = {\lbrack\left. R \middle| p \right.\rbrack}$. For their
world-to-camera counterparts
$P^{\prime} = {\lbrack\left. R^{\prime} \middle| p^{\prime} \right.\rbrack} = {\lbrack\left. R^{\top} \middle| {- {R^{\top}p}} \right.\rbrack}$,
we can first change [[Eq.]{.ltx_text .ltx_ref_tag} [13]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A1.E13 "In 0.A.2 Transforming Gaussians from world to pixel coordinates ‣ Appendix 0.A Method Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
to ${\hat{\mu}}_{i,j} = {{R_{i}^{\prime}\mu_{j}} + p^{\prime}}$ and then
write

Report issue for preceding element

  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -- -----------------------------------------------------
     $${\frac{\partial C_{i}}{\partial\mu_{j}} \approx {\frac{\partial C_{i}}{\partial{\hat{\mu}}_{i,j}}\frac{\partial{\hat{\mu}}_{i,j}}{\partial\mu_{j}}} = {\frac{\partial C_{i}}{\partial{\hat{\mu}}_{i,j}}R_{i}^{\prime}}},{\frac{\partial{\hat{\mu}}_{i,j}}{\partial p_{i}^{\prime}} = I = {R_{i}^{\prime}{(R_{i}^{\prime})}^{\top}} = {\frac{\partial{\hat{\mu}}_{i,j}}{\partial\mu_{j}}{(R_{i}^{\prime})}^{\top}}}$$      [(18)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -- -----------------------------------------------------

and, consequently

Report issue for preceding element

  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----------------------------------------------------
     $$\frac{\partial C_{i}}{\partial p_{i}} \approx {\sum\limits_{j}{\frac{\partial C_{i}}{\partial{\hat{\mu}}_{i,j}}\frac{\partial{\hat{\mu}}_{i,j}}{\partial p_{i}^{\prime}}}} = {\sum\limits_{j}{\frac{\partial C_{i}}{\partial{\hat{\mu}}_{i,j}}\frac{\partial{\hat{\mu}}_{i,j}}{\mu_{j}}{(R_{i}^{\prime})}^{\top}}} \approx {\sum\limits_{j}{\frac{\partial C_{i}}{\partial\mu_{j}}{(R_{i}^{\prime})}^{\top}}}$$      [(19)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----------------------------------------------------

and

Report issue for preceding element

  -- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----------------------------------------------------
     $${\frac{\partial C_{i}}{\partial\nu} \approx {\sum\limits_{j}{\frac{\partial C_{i}}{\partial\mu_{j}}{(R_{i}^{\prime})}^{\top}\frac{\partial R_{i}^{\prime}}{\partial\nu}\mu_{j}}}}.$$      [(20)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----------------------------------------------------

The latter format in [[Eq.]{.ltx_text .ltx_ref_tag} [19]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A1.E19 "In 0.A.3 Differentiation with respect to the camera pose ‣ Appendix 0.A Method Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
and [[Eq.]{.ltx_text .ltx_ref_tag} [20]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A1.E20 "In 0.A.3 Differentiation with respect to the camera pose ‣ Appendix 0.A Method Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
is used in our implementation.

Report issue for preceding element
:::
:::

::: {#Pt0.A1.SS4 .section .ltx_subsection}
### [0.A.4 ]{.ltx_tag .ltx_tag_subsection}Derivation of the pixel velocity formula {#a.4-derivation-of-the-pixel-velocity-formula .ltx_title .ltx_title_subsection}

Report issue for preceding element

::: {#Pt0.A1.SS4.p1 .ltx_para}
The derivative of the pixel coordinates $\mu^{\prime}$ of a Gaussian
center with respect to camera motion $P{(t)}$ is

Report issue for preceding element

  -- ---------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ --
     $\frac{d}{dt}\mu^{\prime}{({P{(t)}})}$   $= {J_{\hat{\mu}\mapsto{\mu^{\prime},t}}\frac{d}{dt}\hat{\mu}{({P{(t)}})}}$                                                                                                                                                          
                                              $= {J_{i}\frac{d}{dt}R_{i}^{\top}{(t)}{({\mu_{j} - {p_{i}{(t)}}})}}$                                                                                                                                                                 
                                              $= {J_{i}\frac{d}{dt}{({R_{i}{\exp{({t{\lbrack\omega_{j}\rbrack}_{\times}})}}})}^{\top}{({\mu_{j} - {({p_{i} + {{t \cdot R_{i}}v_{i}}})}})}}$                                                                                        
                                              $= J_{i}{({({\lbrack\omega_{j}\rbrack}_{\times}^{\top}\exp{(t{\lbrack\omega_{j}\rbrack}_{\times})}^{\top}R_{i}^{\top})}{(\mu_{j} - p_{i}{(t)})} - \exp{(t{\lbrack\omega_{j}\rbrack}_{\times})}^{\top}R_{i}^{\top}{(R_{i}v_{i})})}$   
                                              $= J_{i}{({\lbrack\omega_{j}\rbrack}_{\times}^{\top}\exp{(t{\lbrack\omega_{j}\rbrack}_{\times})}^{\top}{\hat{\mu}}_{i,j} - \exp{(t{\lbrack\omega_{j}\rbrack}_{\times})}^{\top}v_{i})}.$                                              
  -- ---------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ --

The derivative at $t = 0$ is

Report issue for preceding element

  -- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     $${{{\frac{d}{dt}\mu^{\prime}{({P{(t)}})}}|}_{t = 0} = {J_{i}{({{{\lbrack\omega_{j}\rbrack}_{\times}^{\top}{\hat{\mu}}_{i,j}} - v_{i}})}} = {- {J_{i}{({{\omega_{j} \times {\hat{\mu}}_{i,j}} + v_{i}})}}} = v_{i,j}^{\prime}}.$$   
  -- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
:::
:::

::: {#Pt0.A1.SS5 .section .ltx_subsection}
### [0.A.5 ]{.ltx_tag .ltx_tag_subsection}Key Frame Selection {#a.5-key-frame-selection .ltx_title .ltx_title_subsection}

Report issue for preceding element

::: {#Pt0.A1.SS5.p1 .ltx_para}
The level of motion blur in a given frame can be estimated using a VIO
or VISLAM system by examining the pixel velocities of the 3D positions
$l_{j}$, $j = {1,\ldots,N_{i}^{lm}}$ of the sparse SLAM landmarks
visible in the frame in question. The Open Source part of the
Spectacular AI Mapping
tools \[[1](https://arxiv.org/html/2403.13327v3#bib.bib1){.ltx_ref}\]
computes a *motion blur score* as

Report issue for preceding element

  -- ---------------------------------------------------------------------------------------------------------------------------------------------------- -- -----------------------------------------------------
     $$M_{i} = {\frac{1}{N_{i}^{lm}}{\sum\limits_{j}\left\| {J_{i}\left( {{\omega_{i} \times {({{R_{i}l_{j}} + p_{i}})}} + v_{i}} \right)} \right\|}}$$      [(21)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- ---------------------------------------------------------------------------------------------------------------------------------------------------- -- -----------------------------------------------------

where $J_{i}$ is as in [[Eq.]{.ltx_text .ltx_ref_tag} [15]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A1.E15 "In 0.A.2 Transforming Gaussians from world to pixel coordinates ‣ Appendix 0.A Method Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref},
$(R_{i},t_{i})$ is the world-to-camera pose and $(v_{i},\omega_{i})$ are
the instantaneous camera-coordinate linear and angular velocities of the
frame, respectively.

Report issue for preceding element
:::

::: {#Pt0.A1.SS5.p2 .ltx_para}
To reduce sporadic motion blur, the Spectacular AI software drops all
key frame candidates $i$ which have the highest blur score in a
neighbourhood of 4 key frame candidates
($\lbrack{i - 2},{i - 1},i,{i + 1}\rbrack$). We utilize the motion blur
score computed using [[Eq.]{.ltx_text .ltx_ref_tag} [21]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A1.E21 "In 0.A.5 Key Frame Selection ‣ Appendix 0.A Method Details ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
also for the training and evaluation subset partitions described in
[[Sec.]{.ltx_text .ltx_ref_tag} [4.2]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S4.SS2 "4.2 Smartphone Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}.

Report issue for preceding element
:::

::: {#Pt0.A1.SS5.p3 .ltx_para}
Additionally, we ran an experiment where the evaluation frames were
picked by selecting every 8th key frame and switched off the motion blur
score based filtering. See [[Table]{.ltx_text
.ltx_ref_tag} [6]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A4.T6 "In Blur-based key frame selection ‣ 0.D.0.1 Alternative intrinsics ‣ Appendix 0.D Additional Results ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}.
The numeric results were very similar with motion blur filtering
enabled. In the main paper, we chose to include the results with motion
blur filtering enabled, to demonstrate that our approach can decrease
motion-blur-induced effects even after the easy options for filtering
blurry input frames have been exhausted. This highlights the usefulness
of deblurring utilizing a 3D image formation model in the context of
differentiable rendering.

Report issue for preceding element
:::
:::

::: {#Pt0.A1.SS6 .section .ltx_subsection}
### [0.A.6 ]{.ltx_tag .ltx_tag_subsection}Transferring velocities from one SLAM method to another {#a.6-transferring-velocities-from-one-slam-method-to-another .ltx_title .ltx_title_subsection}

Report issue for preceding element

::: {#Pt0.A1.SS6.p1 .ltx_para}
Assuming we have a matching set of $N$ poses $P_{i} = {(R_{i},p_{i})}$
for two methods (COLMAP and SAI), which differ by a ${Sim}{(3)}$
transformation, we can map the linear frame velocities as

Report issue for preceding element

  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------ -- -----------------------------------------------------
     $${{v_{i} = {\frac{s{(p^{COLMAP})}}{s{(p^{SAI})}}v_{i}^{SAI}}},{{s^{2}{(p)}}:={\sum\limits_{i}\left\| {p_{i} - \frac{\sum_{i}p_{i}}{N}} \right\|^{2}}}}.$$      [(22)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------ -- -----------------------------------------------------

Note that the result does not depend on the rotation or translation part
of the transformation and the accuracy of the result depends on the
average scale consistency of each method.

Report issue for preceding element
:::
:::
:::

::: {#Pt0.A2 .section .ltx_appendix}
[Appendix 0.B ]{.ltx_tag .ltx_tag_appendix}Data Sets and Metrics {#appendix-0.b-data-sets-and-metrics .ltx_title .ltx_title_appendix}
----------------------------------------------------------------

Report issue for preceding element

::: {#Pt0.A2.SS0.SSS0.Px1 .section .ltx_paragraph}
##### Modifications to the Deblur-NeRF data set {#modifications-to-the-deblur-nerf-data-set .ltx_title .ltx_title_paragraph}

Report issue for preceding element

::: {#Pt0.A2.SS0.SSS0.Px1.p1 .ltx_para}
The original version of the Deblur-NeRF data set
\[[25](https://arxiv.org/html/2403.13327v3#bib.bib25){.ltx_ref}\] (or
the BAD-NeRF
re-render \[[40](https://arxiv.org/html/2403.13327v3#bib.bib40){.ltx_ref}\])
did not include the velocity data, which was randomly generated by the
original authors with an unspecified seed number. We regenerated the
images using the Blender
\[[4](https://arxiv.org/html/2403.13327v3#bib.bib4){.ltx_ref}\] files,
which we modified by fixing the random seed, and also included velocity
information in our outputs. All of the models mentioned in the
\[[25](https://arxiv.org/html/2403.13327v3#bib.bib25){.ltx_ref}\] were
also not available (*e.g*.[]{#Pt0.A2.SS0.SSS0.Px1.p1.1.2 .ltx_text}
'trolley'), and those cases are omitted from our version of the data
set.

Report issue for preceding element
:::

::: {#Pt0.A2.SS0.SSS0.Px1.p2 .ltx_para}
As in \[[25](https://arxiv.org/html/2403.13327v3#bib.bib25){.ltx_ref}\],
the evaluation images in the data set were rendered without blur or
rollings shutter effects, which allows us to assess the novel-view
synthesis performance of the underlying 'sharp' reconstruction
independent of the accuracy of our simulation and compensation for these
effects.

Report issue for preceding element
:::

::: {#Pt0.A2.SS0.SSS0.Px1.p3 .ltx_para}
Furthermore, we switched from the custom 10-sample blur implementation
to Blender's built-in motion blur and rolling-shutter effects, which are
less prone to sampling artefacts. In addition, we adjusted the caustics
rendering settings in the Blender scenes to reduce non-deterministic
raytracing sampling noise in the data with reasonable sample counts. As
a result, our modified data set, published as supplementary material,
provides effectively deterministic rendering capabilities and improves
the reproducibility of the results compared to the original version. Our
modified data set also includes de-focus blur, which was studied
in \[[25](https://arxiv.org/html/2403.13327v3#bib.bib25){.ltx_ref}\] but
not in this paper, which is why this variation is also omitted
from [[Table]{.ltx_text .ltx_ref_tag} [1]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S4.T1 "In 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}.

Report issue for preceding element
:::

::: {#Pt0.A2.SS0.SSS0.Px1.p4 .ltx_para}
We also note that the different versions of the Deblur-NeRF data set
have different gamma correction factors. The original
version \[[25](https://arxiv.org/html/2403.13327v3#bib.bib25){.ltx_ref}\]
and our re-render use $\gamma = 2.2$ while the
BAD-NeRF \[[40](https://arxiv.org/html/2403.13327v3#bib.bib40){.ltx_ref}\]
re-render use $\gamma = 1$. In [[Sec.]{.ltx_text
.ltx_ref_tag} [4.1]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S4.SS1 "4.1 Synthetic Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref},
we modify the $\gamma$ parameter of our method accordingly.

Report issue for preceding element
:::

::: {#Pt0.A2.SS0.SSS0.Px1.p5 .ltx_para}
Finally, we observed that the *Tanabata* scene in the BAD-NeRF re-render
contains an error that manifests as lower metrics: the sharp evaluation
images and blurry training images have been rendered using a slightly
different 3D model, apparently unintentionally. This issue is not
present in the original version in
\[[25](https://arxiv.org/html/2403.13327v3#bib.bib25){.ltx_ref}\].

Report issue for preceding element
:::
:::
:::

::: {#Pt0.A3 .section .ltx_appendix}
[Appendix 0.C ]{.ltx_tag .ltx_tag_appendix}Experiment Details {#appendix-0.c-experiment-details .ltx_title .ltx_title_appendix}
-------------------------------------------------------------

Report issue for preceding element

::: {#Pt0.A3.SS0.SSS0.Px1 .section .ltx_paragraph}
##### Splatfacto hyperparameters {#splatfacto-hyperparameters .ltx_title .ltx_title_paragraph}

Report issue for preceding element

::: {#Pt0.A3.SS0.SSS0.Px1.p1 .ltx_para}
The baseline method in our tests is Splatfacto in Nerfstudio version
1.1.0, which includes several improvements from the recent works
\[[48](https://arxiv.org/html/2403.13327v3#bib.bib48){.ltx_ref},
[49](https://arxiv.org/html/2403.13327v3#bib.bib49){.ltx_ref},
[43](https://arxiv.org/html/2403.13327v3#bib.bib43){.ltx_ref}\]. For the
[gsplat]{#Pt0.A3.SS0.SSS0.Px1.p1.1.1 .ltx_text .ltx_font_typewriter}
code, we used the main branch version accessed on 2024-05-07 and our
method has also been implemented based on these versions.

Report issue for preceding element
:::

::: {#Pt0.A3.SS0.SSS0.Px1.p2 .ltx_para}
The following non-default parameters were used for both the Splatfacto
baseline and our method: [(i)]{#Pt0.A3.SS0.SSS0.Px1.p2.1.1 .ltx_text
.ltx_font_italic} limited iteration count to 20k.
[(ii)]{#Pt0.A3.SS0.SSS0.Px1.p2.1.2 .ltx_text .ltx_font_italic} enabled
antialiased rasterization; (*cf*.[]{#Pt0.A3.SS0.SSS0.Px1.p2.1.4
.ltx_text} \[[49](https://arxiv.org/html/2403.13327v3#bib.bib49){.ltx_ref}\]);
[(iii)]{#Pt0.A3.SS0.SSS0.Px1.p2.1.5 .ltx_text .ltx_font_italic} enabled
scale regularization (*cf*.[]{#Pt0.A3.SS0.SSS0.Px1.p2.1.7
.ltx_text} \[[43](https://arxiv.org/html/2403.13327v3#bib.bib43){.ltx_ref}\]).
Furthermore, for smartphone data where our method was used with
$\gamma = 2.2$, the minimum RGB level of all color channels was set to
10 in the training data, to avoid large negative logit color values,
which resulted in artefacts in the proximity of dark areas in certain
scenes. This modification was only used in our method and not the
baseline Splatfacto.

Report issue for preceding element
:::
:::
:::

::: {#Pt0.A4 .section .ltx_appendix}
[Appendix 0.D ]{.ltx_tag .ltx_tag_appendix}Additional Results {#appendix-0.d-additional-results .ltx_title .ltx_title_appendix}
-------------------------------------------------------------

Report issue for preceding element

::: {#Pt0.A4.SS0.SSS0.Px1 .section .ltx_paragraph}
##### Figures {#figures .ltx_title .ltx_title_paragraph}

Report issue for preceding element

::: {#Pt0.A4.SS0.SSS0.Px1.p1 .ltx_para}
The synthetic data results for the remaining scenes are visualized in
[[Fig.]{.ltx_text .ltx_ref_tag} [6]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A4.F6 "In Figures ‣ Appendix 0.D Additional Results ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
and [[Fig.]{.ltx_text .ltx_ref_tag} [7]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A4.F7 "In Figures ‣ Appendix 0.D Additional Results ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}.
The reconstructions for rest of the smartphone data sets presented in
[[Table]{.ltx_text .ltx_ref_tag} [3]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S4.T3 "In Re-rendered data ‣ 4.1 Synthetic Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
are shown in [[Fig.]{.ltx_text .ltx_ref_tag} [9]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A4.F9 "In Blur-based key frame selection ‣ 0.D.0.1 Alternative intrinsics ‣ Appendix 0.D Additional Results ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}.

Report issue for preceding element
:::

![[[Figure 6]{#Pt0.A4.F6.6.1.1 .ltx_text style="font-size:129%;"}:
]{.ltx_tag .ltx_tag_figure}[3DGS reconstructions from the synthetic
[tanabata]{#Pt0.A4.F6.7.2.1 .ltx_text .ltx_font_typewriter}
scene]{#Pt0.A4.F6.7.2 .ltx_text
style="font-size:129%;"}](3dgs_deblur_paper_files/synthetic-tanabata-mb-baseline_JJYT.jpg){#Pt0.A4.F6.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.g1
.ltx_graphics .ltx_img_landscape width="202" height="135"}

Report issue for preceding element

![[[Figure 7]{#Pt0.A4.F7.6.1.1 .ltx_text style="font-size:129%;"}:
]{.ltx_tag .ltx_tag_figure}[3DGS reconstructions from the synthetic
[pool]{#Pt0.A4.F7.7.2.1 .ltx_text .ltx_font_typewriter}
scene]{#Pt0.A4.F7.7.2 .ltx_text
style="font-size:129%;"}](3dgs_deblur_paper_files/synthetic-pool-mb-baseline_JJYT.jpg){#Pt0.A4.F7.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.g1
.ltx_graphics .ltx_img_landscape width="202" height="135"}

Report issue for preceding element
:::

::: {#Pt0.A4.SS0.SSS1 .section .ltx_subsubsection}
#### [0.D.0.1 ]{.ltx_tag .ltx_tag_subsubsection}Alternative intrinsics {#d.0.1-alternative-intrinsics .ltx_title .ltx_title_subsubsection}

Report issue for preceding element

::: {#Pt0.A4.SS0.SSS1.p1 .ltx_para}
In addition to the main sequences, we recorded separate manual camera
calibration data for smart phones, using the same fixed focus distance
as in the other sequences, and used the Kalibr software package
(<https://github.com/ethz-asl/kalibr>) to compute a accurate intrinsic
calibration parameters for those devices. We also recorded built-in
calibration data reported by the devices.

Report issue for preceding element
:::

::: {#Pt0.A4.SS0.SSS1.p2 .ltx_para}
We then used then trained our method with data where the
COLMAP-calibrated intrinsics have been replaced by the manually
calibrated intrinsics or the build-in calibration data. The results are
presented in [[Table]{.ltx_text .ltx_ref_tag} [5]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A4.T5 "In Blur-based key frame selection ‣ 0.D.0.1 Alternative intrinsics ‣ Appendix 0.D Additional Results ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}.
The built-in calibration data reduced the accuracy in all cases, but the
manual calibration improved them for some Android data. We suspect that
the reason behind this is that COLMAP also utilizes the camera intrinsic
parameters to compensate also for the rolling shutter effect, which
results to unoptimal intrinsics solution for a method which can also
perform rolling shutter compensation.

Report issue for preceding element
:::

::: {#Pt0.A4.SS0.SSS1.p3 .ltx_para}
We also note that COLMAP-estimated poses do not generally represent the
best solution for a given (different) set of camera intrinsics, since
slightly inaccurate intrinsics may be compensated by adjusting the
poses. Even though this effect can be compensated by our pose
optimization, it still appears that for the iPhone data where the
rolling shutter effect is negligible, COLMAP-calibrated intrinsics,
which are optimized per session, yield better results than manual
calibration, which may also have its own inaccuracies.

Report issue for preceding element
:::

::: {#Pt0.A4.SS0.SSS1.Px1 .section .ltx_paragraph}
##### Blur-based key frame selection {#blur-based-key-frame-selection .ltx_title .ltx_title_paragraph}

Report issue for preceding element

::: {#Pt0.A4.SS0.SSS1.Px1.p1 .ltx_para}
[[Table]{.ltx_text .ltx_ref_tag} [6]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#Pt0.A4.T6 "In Blur-based key frame selection ‣ 0.D.0.1 Alternative intrinsics ‣ Appendix 0.D Additional Results ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
shows the results for an experiment where all motion-blur-based scoring
for training-evaluation split and key frame selection have been
disabled. The results are relatively similar to those in
[[Table]{.ltx_text .ltx_ref_tag} [3]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S4.T3 "In Re-rendered data ‣ 4.1 Synthetic Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}.
However, with more blurry evaluation frames, it becomes less clear if
the metrics reward sharp reconstructions or accurate simulation of
motion blur and rolling shutter effects in the forward model.

Report issue for preceding element
:::

[[Table 5]{#Pt0.A4.T5.14.1.1 .ltx_text style="font-size:129%;"}:
]{.ltx_tag .ltx_tag_table}[Smartphone results with alternative
calibration methods. The COLMAP variant data matches 'Ours' in
[[Table]{.ltx_text .ltx_ref_tag} [3]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S4.T3 "In Re-rendered data ‣ 4.1 Synthetic Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}.]{#Pt0.A4.T5.15.2
.ltx_text style="font-size:129%;"}
:::
:::
:::

[COLMAP intrinsics]{#Pt0.A4.T5.9.10.1.2.1 .ltx_text
style="font-size:70%;"}

[Built-in calibration]{#Pt0.A4.T5.9.10.1.3.1 .ltx_text
style="font-size:70%;"}

[Manual calibration]{#Pt0.A4.T5.9.10.1.4.1 .ltx_text
style="font-size:70%;"}

[PSNR$\uparrow$]{#Pt0.A4.T5.1.1.1.1 .ltx_text style="font-size:50%;"}

[SSIM$\uparrow$]{#Pt0.A4.T5.2.2.2.1 .ltx_text style="font-size:50%;"}

[LPIPS$\downarrow$]{#Pt0.A4.T5.3.3.3.1 .ltx_text style="font-size:50%;"}

[PSNR$\uparrow$]{#Pt0.A4.T5.4.4.4.1 .ltx_text style="font-size:50%;"}

[SSIM$\uparrow$]{#Pt0.A4.T5.5.5.5.1 .ltx_text style="font-size:50%;"}

[LPIPS$\downarrow$]{#Pt0.A4.T5.6.6.6.1 .ltx_text style="font-size:50%;"}

[PSNR$\uparrow$]{#Pt0.A4.T5.7.7.7.1 .ltx_text style="font-size:50%;"}

[SSIM$\uparrow$]{#Pt0.A4.T5.8.8.8.1 .ltx_text style="font-size:50%;"}

[LPIPS$\downarrow$]{#Pt0.A4.T5.9.9.9.1 .ltx_text style="font-size:50%;"}

[iphone-lego1]{#Pt0.A4.T5.9.11.1.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[29.20]{#Pt0.A4.T5.9.11.1.2.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.933]{#Pt0.A4.T5.9.11.1.3.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.116]{#Pt0.A4.T5.9.11.1.4.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[27.37]{#Pt0.A4.T5.9.11.1.5.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.903]{#Pt0.A4.T5.9.11.1.6.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.140]{#Pt0.A4.T5.9.11.1.7.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[27.37]{#Pt0.A4.T5.9.11.1.8.1 .ltx_text style="font-size:70%;"}

[.902]{#Pt0.A4.T5.9.11.1.9.1 .ltx_text style="font-size:70%;"}

[.143]{#Pt0.A4.T5.9.11.1.10.1 .ltx_text style="font-size:70%;"}

[iphone-lego2]{#Pt0.A4.T5.9.12.2.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[27.95]{#Pt0.A4.T5.9.12.2.2.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.926]{#Pt0.A4.T5.9.12.2.3.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.117]{#Pt0.A4.T5.9.12.2.4.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[27.88]{#Pt0.A4.T5.9.12.2.5.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.917]{#Pt0.A4.T5.9.12.2.6.1 .ltx_text style="font-size:70%;"}

[.124]{#Pt0.A4.T5.9.12.2.7.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[27.03]{#Pt0.A4.T5.9.12.2.8.1 .ltx_text style="font-size:70%;"}

[.917]{#Pt0.A4.T5.9.12.2.9.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.124]{#Pt0.A4.T5.9.12.2.10.1 .ltx_text style="font-size:70%;"}

[iphone-lego3]{#Pt0.A4.T5.9.13.3.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[24.50]{#Pt0.A4.T5.9.13.3.2.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.824]{#Pt0.A4.T5.9.13.3.3.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.203]{#Pt0.A4.T5.9.13.3.4.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[23.58]{#Pt0.A4.T5.9.13.3.5.1 .ltx_text style="font-size:70%;"}

[.759]{#Pt0.A4.T5.9.13.3.6.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.250]{#Pt0.A4.T5.9.13.3.7.1 .ltx_text style="font-size:70%;"}

[23.83]{#Pt0.A4.T5.9.13.3.8.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.758]{#Pt0.A4.T5.9.13.3.9.1 .ltx_text style="font-size:70%;"}

[.238]{#Pt0.A4.T5.9.13.3.10.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[iphone-pots1]{#Pt0.A4.T5.9.14.4.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[29.10]{#Pt0.A4.T5.9.14.4.2.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.926]{#Pt0.A4.T5.9.14.4.3.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.135]{#Pt0.A4.T5.9.14.4.4.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[28.21]{#Pt0.A4.T5.9.14.4.5.1 .ltx_text style="font-size:70%;"}

[.885]{#Pt0.A4.T5.9.14.4.6.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.171]{#Pt0.A4.T5.9.14.4.7.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[28.25]{#Pt0.A4.T5.9.14.4.8.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.883]{#Pt0.A4.T5.9.14.4.9.1 .ltx_text style="font-size:70%;"}

[.177]{#Pt0.A4.T5.9.14.4.10.1 .ltx_text style="font-size:70%;"}

[iphone-pots2]{#Pt0.A4.T5.9.15.5.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[28.00]{#Pt0.A4.T5.9.15.5.2.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.882]{#Pt0.A4.T5.9.15.5.3.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.180]{#Pt0.A4.T5.9.15.5.4.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[27.53]{#Pt0.A4.T5.9.15.5.5.1 .ltx_text style="font-size:70%;"}

[.851]{#Pt0.A4.T5.9.15.5.6.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.198]{#Pt0.A4.T5.9.15.5.7.1 .ltx_text style="font-size:70%;"}

[27.68]{#Pt0.A4.T5.9.15.5.8.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.848]{#Pt0.A4.T5.9.15.5.9.1 .ltx_text style="font-size:70%;"}

[.197]{#Pt0.A4.T5.9.15.5.10.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[pixel5-lamp]{#Pt0.A4.T5.9.16.6.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[30.46]{#Pt0.A4.T5.9.16.6.2.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.833]{#Pt0.A4.T5.9.16.6.3.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.196]{#Pt0.A4.T5.9.16.6.4.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[29.50]{#Pt0.A4.T5.9.16.6.5.1 .ltx_text style="font-size:70%;"}

[.805]{#Pt0.A4.T5.9.16.6.6.1 .ltx_text style="font-size:70%;"}

[.225]{#Pt0.A4.T5.9.16.6.7.1 .ltx_text style="font-size:70%;"}

[30.45]{#Pt0.A4.T5.9.16.6.8.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.830]{#Pt0.A4.T5.9.16.6.9.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.199]{#Pt0.A4.T5.9.16.6.10.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[pixel5-plant]{#Pt0.A4.T5.9.17.7.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[27.90]{#Pt0.A4.T5.9.17.7.2.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.920]{#Pt0.A4.T5.9.17.7.3.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.204]{#Pt0.A4.T5.9.17.7.4.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[27.14]{#Pt0.A4.T5.9.17.7.5.1 .ltx_text style="font-size:70%;"}

[.902]{#Pt0.A4.T5.9.17.7.6.1 .ltx_text style="font-size:70%;"}

[.239]{#Pt0.A4.T5.9.17.7.7.1 .ltx_text style="font-size:70%;"}

[27.54]{#Pt0.A4.T5.9.17.7.8.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.919]{#Pt0.A4.T5.9.17.7.9.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.205]{#Pt0.A4.T5.9.17.7.10.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[pixel5-table]{#Pt0.A4.T5.9.18.8.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[31.86]{#Pt0.A4.T5.9.18.8.2.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.919]{#Pt0.A4.T5.9.18.8.3.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.196]{#Pt0.A4.T5.9.18.8.4.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[30.60]{#Pt0.A4.T5.9.18.8.5.1 .ltx_text style="font-size:70%;"}

[.883]{#Pt0.A4.T5.9.18.8.6.1 .ltx_text style="font-size:70%;"}

[.221]{#Pt0.A4.T5.9.18.8.7.1 .ltx_text style="font-size:70%;"}

[32.26]{#Pt0.A4.T5.9.18.8.8.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.919]{#Pt0.A4.T5.9.18.8.9.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.194]{#Pt0.A4.T5.9.18.8.10.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[s20-bike]{#Pt0.A4.T5.9.19.9.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[28.93]{#Pt0.A4.T5.9.19.9.2.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.911]{#Pt0.A4.T5.9.19.9.3.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.242]{#Pt0.A4.T5.9.19.9.4.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[28.15]{#Pt0.A4.T5.9.19.9.5.1 .ltx_text style="font-size:70%;"}

[.887]{#Pt0.A4.T5.9.19.9.6.1 .ltx_text style="font-size:70%;"}

[.295]{#Pt0.A4.T5.9.19.9.7.1 .ltx_text style="font-size:70%;"}

[29.35]{#Pt0.A4.T5.9.19.9.8.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.909]{#Pt0.A4.T5.9.19.9.9.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.246]{#Pt0.A4.T5.9.19.9.10.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[s20-bikerack]{#Pt0.A4.T5.9.20.10.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[29.74]{#Pt0.A4.T5.9.20.10.2.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.831]{#Pt0.A4.T5.9.20.10.3.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.221]{#Pt0.A4.T5.9.20.10.4.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[27.62]{#Pt0.A4.T5.9.20.10.5.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.814]{#Pt0.A4.T5.9.20.10.6.1 .ltx_text style="font-size:70%;"}

[.269]{#Pt0.A4.T5.9.20.10.7.1 .ltx_text style="font-size:70%;"}

[27.26]{#Pt0.A4.T5.9.20.10.8.1 .ltx_text style="font-size:70%;"}

[.827]{#Pt0.A4.T5.9.20.10.9.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.234]{#Pt0.A4.T5.9.20.10.10.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[s20-sign]{#Pt0.A4.T5.9.21.11.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[26.84]{#Pt0.A4.T5.9.21.11.2.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.812]{#Pt0.A4.T5.9.21.11.3.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.206]{#Pt0.A4.T5.9.21.11.4.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[25.31]{#Pt0.A4.T5.9.21.11.5.1 .ltx_text style="font-size:70%;"}

[.773]{#Pt0.A4.T5.9.21.11.6.1 .ltx_text style="font-size:70%;"}

[.249]{#Pt0.A4.T5.9.21.11.7.1 .ltx_text style="font-size:70%;"}

[26.84]{#Pt0.A4.T5.9.21.11.8.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.810]{#Pt0.A4.T5.9.21.11.9.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.206]{#Pt0.A4.T5.9.21.11.10.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[average]{#Pt0.A4.T5.9.22.12.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[28.59]{#Pt0.A4.T5.9.22.12.2.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.883]{#Pt0.A4.T5.9.22.12.3.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.183]{#Pt0.A4.T5.9.22.12.4.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[27.54]{#Pt0.A4.T5.9.22.12.5.1 .ltx_text style="font-size:70%;"}

[.853]{#Pt0.A4.T5.9.22.12.6.1 .ltx_text style="font-size:70%;"}

[.217]{#Pt0.A4.T5.9.22.12.7.1 .ltx_text style="font-size:70%;"}

[27.99]{#Pt0.A4.T5.9.22.12.8.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.866]{#Pt0.A4.T5.9.22.12.9.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

[.197]{#Pt0.A4.T5.9.22.12.10.1 .ltx_text .ltx_font_italic
style="font-size:70%;"}

Report issue for preceding element

[[Table 6]{#Pt0.A4.T6.11.1.1 .ltx_text style="font-size:129%;"}:
]{.ltx_tag .ltx_tag_table}[Results corresponding to [[Table]{.ltx_text
.ltx_ref_tag} [3]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S4.T3 "In Re-rendered data ‣ 4.1 Synthetic Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
with motion-blur-based key frame selection and training-evaluation split
disabled.]{#Pt0.A4.T6.12.2 .ltx_text style="font-size:129%;"}

[Splatfacto]{#Pt0.A4.T6.6.7.1.2.1 .ltx_text style="font-size:70%;"}

[Ours]{#Pt0.A4.T6.6.7.1.3.1 .ltx_text style="font-size:70%;"}

[PSNR$\uparrow$]{#Pt0.A4.T6.1.1.1.1 .ltx_text style="font-size:50%;"}

[SSIM$\uparrow$]{#Pt0.A4.T6.2.2.2.1 .ltx_text style="font-size:50%;"}

[LPIPS$\downarrow$]{#Pt0.A4.T6.3.3.3.1 .ltx_text style="font-size:50%;"}

[PSNR$\uparrow$]{#Pt0.A4.T6.4.4.4.1 .ltx_text style="font-size:50%;"}

[SSIM$\uparrow$]{#Pt0.A4.T6.5.5.5.1 .ltx_text style="font-size:50%;"}

[LPIPS$\downarrow$]{#Pt0.A4.T6.6.6.6.1 .ltx_text style="font-size:50%;"}

[iphone-lego1]{#Pt0.A4.T6.6.8.1.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[26.26]{#Pt0.A4.T6.6.8.1.2.1 .ltx_text style="font-size:70%;"}

[.892]{#Pt0.A4.T6.6.8.1.3.1 .ltx_text style="font-size:70%;"}

[.202]{#Pt0.A4.T6.6.8.1.4.1 .ltx_text style="font-size:70%;"}

[27.48]{#Pt0.A4.T6.6.8.1.5.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.930]{#Pt0.A4.T6.6.8.1.6.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.152]{#Pt0.A4.T6.6.8.1.7.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[iphone-lego2]{#Pt0.A4.T6.6.9.2.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[27.44]{#Pt0.A4.T6.6.9.2.2.1 .ltx_text style="font-size:70%;"}

[.914]{#Pt0.A4.T6.6.9.2.3.1 .ltx_text style="font-size:70%;"}

[.153]{#Pt0.A4.T6.6.9.2.4.1 .ltx_text style="font-size:70%;"}

[27.76]{#Pt0.A4.T6.6.9.2.5.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.933]{#Pt0.A4.T6.6.9.2.6.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.140]{#Pt0.A4.T6.6.9.2.7.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[iphone-lego3]{#Pt0.A4.T6.6.10.3.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[24.16]{#Pt0.A4.T6.6.10.3.2.1 .ltx_text style="font-size:70%;"}

[.777]{#Pt0.A4.T6.6.10.3.3.1 .ltx_text style="font-size:70%;"}

[.324]{#Pt0.A4.T6.6.10.3.4.1 .ltx_text style="font-size:70%;"}

[25.63]{#Pt0.A4.T6.6.10.3.5.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.853]{#Pt0.A4.T6.6.10.3.6.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.198]{#Pt0.A4.T6.6.10.3.7.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[iphone-pots1]{#Pt0.A4.T6.6.11.4.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[28.39]{#Pt0.A4.T6.6.11.4.2.1 .ltx_text style="font-size:70%;"}

[.917]{#Pt0.A4.T6.6.11.4.3.1 .ltx_text style="font-size:70%;"}

[.202]{#Pt0.A4.T6.6.11.4.4.1 .ltx_text style="font-size:70%;"}

[28.62]{#Pt0.A4.T6.6.11.4.5.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.938]{#Pt0.A4.T6.6.11.4.6.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.159]{#Pt0.A4.T6.6.11.4.7.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[iphone-pots2]{#Pt0.A4.T6.6.12.5.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[28.84]{#Pt0.A4.T6.6.12.5.2.1 .ltx_text style="font-size:70%;"}

[.878]{#Pt0.A4.T6.6.12.5.3.1 .ltx_text style="font-size:70%;"}

[.280]{#Pt0.A4.T6.6.12.5.4.1 .ltx_text style="font-size:70%;"}

[29.37]{#Pt0.A4.T6.6.12.5.5.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.898]{#Pt0.A4.T6.6.12.5.6.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.210]{#Pt0.A4.T6.6.12.5.7.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[pixel5-lamp]{#Pt0.A4.T6.6.13.6.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[28.36]{#Pt0.A4.T6.6.13.6.2.1 .ltx_text style="font-size:70%;"}

[.865]{#Pt0.A4.T6.6.13.6.3.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.336]{#Pt0.A4.T6.6.13.6.4.1 .ltx_text style="font-size:70%;"}

[31.57]{#Pt0.A4.T6.6.13.6.5.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.845]{#Pt0.A4.T6.6.13.6.6.1 .ltx_text style="font-size:70%;"}

[.193]{#Pt0.A4.T6.6.13.6.7.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[pixel5-plant]{#Pt0.A4.T6.6.14.7.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[26.76]{#Pt0.A4.T6.6.14.7.2.1 .ltx_text style="font-size:70%;"}

[.933]{#Pt0.A4.T6.6.14.7.3.1 .ltx_text style="font-size:70%;"}

[.218]{#Pt0.A4.T6.6.14.7.4.1 .ltx_text style="font-size:70%;"}

[28.55]{#Pt0.A4.T6.6.14.7.5.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.947]{#Pt0.A4.T6.6.14.7.6.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.182]{#Pt0.A4.T6.6.14.7.7.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[pixel5-table]{#Pt0.A4.T6.6.15.8.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[28.39]{#Pt0.A4.T6.6.15.8.2.1 .ltx_text style="font-size:70%;"}

[.916]{#Pt0.A4.T6.6.15.8.3.1 .ltx_text style="font-size:70%;"}

[.240]{#Pt0.A4.T6.6.15.8.4.1 .ltx_text style="font-size:70%;"}

[31.14]{#Pt0.A4.T6.6.15.8.5.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.940]{#Pt0.A4.T6.6.15.8.6.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.185]{#Pt0.A4.T6.6.15.8.7.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[s20-bike]{#Pt0.A4.T6.6.16.9.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[26.81]{#Pt0.A4.T6.6.16.9.2.1 .ltx_text style="font-size:70%;"}

[.904]{#Pt0.A4.T6.6.16.9.3.1 .ltx_text style="font-size:70%;"}

[.288]{#Pt0.A4.T6.6.16.9.4.1 .ltx_text style="font-size:70%;"}

[31.64]{#Pt0.A4.T6.6.16.9.5.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.925]{#Pt0.A4.T6.6.16.9.6.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.238]{#Pt0.A4.T6.6.16.9.7.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[s20-bikerack]{#Pt0.A4.T6.6.17.10.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[26.66]{#Pt0.A4.T6.6.17.10.2.1 .ltx_text style="font-size:70%;"}

[.898]{#Pt0.A4.T6.6.17.10.3.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.258]{#Pt0.A4.T6.6.17.10.4.1 .ltx_text style="font-size:70%;"}

[30.30]{#Pt0.A4.T6.6.17.10.5.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.851]{#Pt0.A4.T6.6.17.10.6.1 .ltx_text style="font-size:70%;"}

[.226]{#Pt0.A4.T6.6.17.10.7.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[s20-sign]{#Pt0.A4.T6.6.18.11.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[22.78]{#Pt0.A4.T6.6.18.11.2.1 .ltx_text style="font-size:70%;"}

[.782]{#Pt0.A4.T6.6.18.11.3.1 .ltx_text style="font-size:70%;"}

[.290]{#Pt0.A4.T6.6.18.11.4.1 .ltx_text style="font-size:70%;"}

[28.57]{#Pt0.A4.T6.6.18.11.5.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.836]{#Pt0.A4.T6.6.18.11.6.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.195]{#Pt0.A4.T6.6.18.11.7.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[average]{#Pt0.A4.T6.6.19.12.1.1 .ltx_text .ltx_font_smallcaps
style="font-size:70%;"}

[26.80]{#Pt0.A4.T6.6.19.12.2.1 .ltx_text style="font-size:70%;"}

[.880]{#Pt0.A4.T6.6.19.12.3.1 .ltx_text style="font-size:70%;"}

[.254]{#Pt0.A4.T6.6.19.12.4.1 .ltx_text style="font-size:70%;"}

[29.15]{#Pt0.A4.T6.6.19.12.5.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.900]{#Pt0.A4.T6.6.19.12.6.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

[.189]{#Pt0.A4.T6.6.19.12.7.1 .ltx_text .ltx_font_bold
style="font-size:70%;"}

Report issue for preceding element

![[[Figure 8]{#Pt0.A4.F8.13.1.1 .ltx_text style="font-size:129%;"}:
]{.ltx_tag .ltx_tag_figure}[Additional smartphone data reconstructions
as in [[Fig.]{.ltx_text .ltx_ref_tag} [4]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S4.F4 "In Re-rendered data ‣ 4.1 Synthetic Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
for the scenes [lego2]{#Pt0.A4.F8.14.2.1 .ltx_text .ltx_font_typewriter}
(iPhone), [plant]{#Pt0.A4.F8.14.2.2 .ltx_text .ltx_font_typewriter}
(Pixel), [lamp]{#Pt0.A4.F8.14.2.3 .ltx_text .ltx_font_typewriter}
(Pixel), [bikerack]{#Pt0.A4.F8.14.2.4 .ltx_text .ltx_font_typewriter}
(S20).]{#Pt0.A4.F8.14.2 .ltx_text
style="font-size:129%;"}](3dgs_deblur_paper_files/colmap-sai-cli-vels-blur-scored-iphone-lego2-baseline_JJYT.jpg){#Pt0.A4.F8.pic1.3.3.3.3.3.3.3.3.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.1.g1
.ltx_graphics .ltx_img_landscape width="144" height="108"}

Report issue for preceding element

![[[Figure 9]{#Pt0.A4.F9.11.1.1 .ltx_text style="font-size:129%;"}:
]{.ltx_tag .ltx_tag_figure}[Additional smartphone data reconstructions
as in [[Fig.]{.ltx_text .ltx_ref_tag} [4]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2403.13327v3#S4.F4 "In Re-rendered data ‣ 4.1 Synthetic Data ‣ 4 Experiments ‣ Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"){.ltx_ref}
for the scenes [lego3]{#Pt0.A4.F9.12.2.1 .ltx_text
.ltx_font_typewriter}, [pots1]{#Pt0.A4.F9.12.2.2 .ltx_text
.ltx_font_typewriter} (iPhone) and [sign]{#Pt0.A4.F9.12.2.3 .ltx_text
.ltx_font_typewriter} (S20).]{#Pt0.A4.F9.12.2 .ltx_text
style="font-size:129%;"}](3dgs_deblur_paper_files/colmap-sai-cli-vels-blur-scored-iphone-lego3-baseline_JJYT.jpg){#Pt0.A4.F9.pic1.3.3.3.3.3.3.3.3.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.1.g1
.ltx_graphics .ltx_img_landscape width="192" height="144"}

Report issue for preceding element

::: {.ltx_pagination .ltx_role_newpage}
:::

Report Issue

::: {#myForm .modal role="dialog" aria-labelledby="modal-title"}
::: {.modal-dialog}
::: {#modal-header .modal-header}
##### Report GitHub Issue {#modal-title .modal-title}
:::

::: {.modal-body}
Title:

Content selection saved. Describe the issue below:

Description:
:::

::: {.modal-footer .d-flex .justify-content-end}
Submit without GitHub

Submit in GitHub
:::
:::
:::

Report Issue for Selection

::: {.ltx_page_footer}
::: {.ltx_page_logo}
Generated by [[ L
[A]{style="font-size: 70%; position: relative; bottom: 2.2pt;"} T
[E]{style="position: relative; bottom: -0.4ex;"}
]{style="letter-spacing: -0.2em; margin-right: 0.1em;"}
[xml]{.ltx_font_smallcaps}
![\[LOGO\]](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](https://math.nist.gov/~BMiller/LaTeXML/){.ltx_LaTeXML_logo}
:::
:::

::: {.keyboard-glossary}
Instructions for reporting errors
---------------------------------

We are continuing to improve HTML versions of papers, and your feedback
helps enhance accessibility and mobile support. To report errors in the
HTML that will help us improve conversion and rendering, choose any of
the methods listed below:

-   Click the \"Report Issue\" button.
-   Open a report feedback form via keyboard, use \"**Ctrl + ?**\".
-   Make a text selection and click the \"Report Issue for Selection\"
    button near your cursor.
-   You can use Alt+Y to toggle on and Alt+Shift+Y to toggle off
    accessible reporting links at each section.

Our team has already identified [the following
issues](https://github.com/arXiv/html_feedback/issues){.ltx_ref}. We
appreciate your time reviewing and reporting rendering errors we may not
have found yet. Your efforts will help us improve the HTML versions for
all readers, because disability should not be a barrier to accessing
research. Thank you for your continued support in championing open
access for all.

Have a free development cycle? Help support accessibility at arXiv! Our
collaborators at LaTeXML maintain a [list of packages that need
conversion](https://github.com/brucemiller/LaTeXML/wiki/Porting-LaTeX-packages-for-LaTeXML){.ltx_ref},
and welcome [developer
contributions](https://github.com/brucemiller/LaTeXML/issues){.ltx_ref}.
:::
