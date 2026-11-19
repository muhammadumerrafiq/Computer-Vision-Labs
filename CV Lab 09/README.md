# FractureVision: Region-Aware Bone Fracture Detection and AI-Assisted Orthopedic Workflow Simulation

**COMSATS University Islamabad, Wah Campus**  
**Department of Computer Science**  
**Author:** Muhammad Umer Rafiq (FA23-BAI-013)  
**Course:** Introduction to Computer Vision (7th Semester)  

---

## 📄 Abstract

Suspected bone fractures are usually examined first with a plain X-ray, but reading the image depends on trained specialists, and fractures can be hard to see when they are thin, partly hidden by overlapping bones, or captured in images of uneven quality. This project presents **FractureVision**, a region-aware system that uses deep learning to detect and localize possible fracture regions in X-ray images and to simulate how this kind of assistance could fit into an orthopedic workflow.

The system uses a YOLO-based object detection model trained on the **FracAtlas** dataset, which contains musculoskeletal radiographs with manually annotated fracture locations. The current version focuses on two anatomical regions: **Hand** and **Leg**. The user selects the region and uploads an X-ray image, and the trained model returns bounding boxes with confidence scores around areas that may contain a fracture. The model has a single detection class (`fracture`), with Hand and Leg serving as user-selected anatomical regions. FractureVision is intended as an accessible academic tool that draws attention to suspicious areas for further clinical review, and is not a replacement for professional medical diagnosis.

**Index Terms:** Bone fracture detection, medical image analysis, deep learning, YOLO, object detection, X-ray imaging, FracAtlas.

---

## 1. Introduction

Suspected bone fractures are usually examined first with a plain X-ray. Digital radiography is widely used for this purpose because it is common and cost-effective, but a diagnosis still depends on an expert reader, which takes time and requires long training [1]. The reading is also not free from error. In a four-year review of diagnostic errors in one accident and emergency department, most of the recorded errors were fractures that had been missed [2]. A missed hand fracture, for example, can lead to delayed treatment and poorer recovery of hand function [3].

Part of the difficulty is that fractures do not all look alike. One may be a clear break, another only a thin line or a small change in the outline of the bone, and how it appears depends on the bone, the viewing angle, and the patient. Image quality varies between scans, and some images even carry text or logos that are not part of the anatomy [1]. Normal structures can also mislead a reader or a model. The ends of bones near the joints of young children can look separated, and the surface of bone in older patients can look rough, both of which may be mistaken for fractures [1]. In the hand, many small bones lie close together and overlap, and tiny fractures near the knuckles and fingertips are easy to overlook [3].

Deep learning has become a common approach in medical image analysis because convolutional networks can learn useful visual patterns directly from labeled images instead of relying on hand-crafted rules [4]. For fractures, Lindsey et al. [5] reported that clinicians who read wrist radiographs with the help of a deep network made fewer misinterpretations than when they worked without it. Many public datasets, however, are limited to classification or lack careful annotation [1]. MURA, for example, provides only binary normal or abnormal labels [6]. For a practitioner reviewing an X-ray, the localized region of a suspected fracture is vastly more actionable than an image-level label.

Object detection models suit this need because they simultaneously predict bounding box spatial coordinates and class probabilities. YOLO (You Only Look Once) frames detection as a single regression problem from full pixels to bounding boxes and class confidences, ensuring rapid inference [7]. Earlier studies have applied detection models to distinct anatomical structures: Thian et al. [8] trained networks on frontal and lateral wrist radiographs, Xue et al. [3] modified Faster R-CNN for hand fractures, and Ju & Cai [9] trained YOLOv8 models on a pediatric wrist dataset with a prototype deployment. However, many prior works relied on private hospital datasets [1], and existing public wrist benchmarks often cover only a single body part.

The **FracAtlas** dataset by Abedeen et al. [1] addresses this scarcity. It contains 4,083 musculoskeletal radiographs across hand, leg, hip, and shoulder regions collected from three hospitals in Bangladesh; 717 images show verified fractures across 922 instances. Annotations include segmentation masks and bounding boxes in YOLO format with a single fracture class. FracAtlas baseline experiments with YOLOv8s yielded recall below 50% on the validation set [1], highlighting the intrinsic difficulty and ample research headroom.

FractureVision strategically targets the **hand** and **leg** regions:
- **Hand:** 1,538 scans (437 fractured), characterized by complex clusters of small overlapping bones.
- **Leg:** 2,272 scans (263 fractured), characterized by elongated structural shafts.
- Combined, they encompass the vast majority of FracAtlas fracture instances (hip and shoulder contain only 63 fractured scans each).

### Key System Contributions
1. **YOLO-Based Fracture Detection:** Deep single-stage detector localized for hand and leg radiographs, trained on FracAtlas with a unified `fracture` class.
2. **Region-Aware Workflow Simulation:** Interactive workflow where users specify the target anatomical region (Hand vs. Leg), upload X-ray imagery, and view highlighted suspicious bounding boxes with predictive confidence intervals.
3. **Clinical Context & Dataset Analysis:** Comprehensive analysis of real-world radiographic challenges, artifact handling, and structural limitations of clinical datasets.

---

## 📚 References

1. I. Abedeen, M. A. Rahman, F. Z. Prottyasha, T. Ahmed, T. M. Chowdhury, and S. Shatabda, *"FracAtlas: A dataset for fracture classification, localization and segmentation of musculoskeletal radiographs,"* Scientific Data, vol. 10, Art. no. 521, 2023. DOI: [10.1038/s41597-023-02432-4](https://doi.org/10.1038/s41597-023-02432-4).
2. H. R. Guly, *"Diagnostic errors in an accident and emergency department,"* Emergency Medicine Journal, vol. 18, no. 4, pp. 263-269, 2001. DOI: [10.1136/emj.18.4.263](https://doi.org/10.1136/emj.18.4.263).
3. L. Xue, W. Yan, P. Luo, X. Zhang, T. Chaikovska, K. Liu, W. Gao, and K. Yang, *"Detection and localization of hand fractures based on GA_Faster R-CNN,"* Alexandria Engineering Journal, vol. 60, no. 5, pp. 4555-4562, 2021. DOI: [10.1016/j.aej.2021.03.005](https://doi.org/10.1016/j.aej.2021.03.005).
4. G. Litjens et al., *"A survey on deep learning in medical image analysis,"* Medical Image Analysis, vol. 42, pp. 60-88, 2017. DOI: [10.1016/j.media.2017.07.005](https://doi.org/10.1016/j.media.2017.07.005).
5. R. Lindsey et al., *"Deep neural network improves fracture detection by clinicians,"* Proceedings of the National Academy of Sciences (PNAS), vol. 115, no. 45, pp. 11591-11596, 2018. DOI: [10.1073/pnas.1806905115](https://doi.org/10.1073/pnas.1806905115).
6. P. Rajpurkar et al., *"MURA: Large dataset for abnormality detection in musculoskeletal radiographs,"* arXiv:1712.06957, 2017.
7. J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, *"You only look once: Unified, real-time object detection,"* in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2016, pp. 779-788. DOI: [10.1109/CVPR.2016.91](https://doi.org/10.1109/CVPR.2016.91).
8. Y. L. Thian et al., *"Convolutional neural networks for automated fracture detection and localization on wrist radiographs,"* Radiology: Artificial Intelligence, vol. 1, no. 1, p. e180001, 2019. DOI: [10.1148/ryai.2019180001](https://doi.org/10.1148/ryai.2019180001).
9. R.-Y. Ju and W. Cai, *"Fracture detection in pediatric wrist trauma X-ray images using YOLOv8 algorithm,"* Scientific Reports, vol. 13, Art. no. 20077, 2023. DOI: [10.1038/s41598-023-47460-7](https://doi.org/10.1038/s41598-023-47460-7).
