# Handwriting Recognition
<!--
Offline Handwritten Text Recognition (HRT) is an active field of research that deals with the transcription of handwritten text contained in images. For humans this is mostly trivial (depending on the handwriting), but there are a number of challenges with the automation of this task. We received two datasets, the Dead Sea Scrolls (DSS) and IAM lines. Both differ in type and quality of the data and therefore require separate approaches. The DSS dataset contains
text in Hebrew which has decayed due to age and is not segmented. Only single letters are given as labeled data, so the text images have to be segmented into single characters which can be then recognized. In the case of the IAM dataset, all initial text has been segmented into lines. These contain text in English and come with transcriptions as labels, so an end-to-end system can be implemented. Methodologies for solving both tasks are presented in this work.
-->

Offline Handwritten Text Recognition (HRT) is a dynamic area of research focused on transcribing handwritten text from images. While humans can often decipher such text with ease (depending on the handwriting), automating this task poses several challenges. In our study, we have access to two distinct datasets: the *Dead Sea Scrolls (DSS)* and *IAM* [[1]](#1) collections. These datasets vary in data type and quality, necessitating separate approaches. 

The DSS dataset comprises aged Hebrew text that has deteriorated over time and is not segmented. However, only individual letters are provided as labeled data, requiring first the segmentation of the original text images into single characters before the recognition process can take place. On the other hand, the IAM dataset features images of handwritten lines of English text accompanied by the respective transcriptions as labels. This allows for the implementation of an end-to-end system. Our work presents methodologies for addressing both tasks, catering to the unique characteristics of each dataset.

The current project was implemented in the context of the course "Handwriting Recognition" taught by Professors [Lambert Schomaker](https://www.ai.rug.nl/~lambert/) and [Maruf A. Dhali](https://www.rug.nl/staff/m.a.dhali/) at [University of Groningen](https://www.rug.nl/).

### 1st Assignment: [Character Segmentation and Recognition on the DSS Collection](https://github.com/ChryssaNab/Handwriting-Recognition/tree/main/character_recognition)
1. [Project Description](https://github.com/ChryssaNab/Handwriting-Recognition/blob/main/HWR_Project_description.pdf) (Task 1 & 2)
2. [Implementation](https://github.com/ChryssaNab/Handwriting-Recognition/tree/main/character_recognition/src)
3. [Report](https://github.com/ChryssaNab/Handwriting-Recognition/blob/main/report/report.pdf) (Section 3)
### 2nd Assignment: [Line Recognition on the IAM Dataset](https://github.com/ChryssaNab/Handwriting-Recognition/tree/main/line_recognition)
1. [Project Description](https://github.com/ChryssaNab/Handwriting-Recognition/blob/main/HWR_Project_description.pdf) (Task 3)
2. [Implementation](https://github.com/ChryssaNab/Handwriting-Recognition/tree/main/line_recognition/src)
3. [Report](https://github.com/ChryssaNab/Handwriting-Recognition/blob/main/report/report.pdf) (Section 4)

---

### References 

<a id="1">[1]</a> 
U. Marti & H. Bunke (2002). The IAM-database: An English sentence database for offline handwriting recognition. *International Journal on Document Analysis and Recognition. vol. 5 (pp. 39-46). DOI: 10.1007/s100320200071*. 

---

### Team

- [Chryssa Nampouri](https://github.com/ChryssaNab)
- Shray Juneja
- Luca Mueller
- Wopke de Vries
