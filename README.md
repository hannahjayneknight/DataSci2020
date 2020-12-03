# Data Science Group Project 2020
## Can School Characteristics Accurately Predict GCSE Results? Can Student Performance be Predicted based on Demographic Information and Past Performance?
###### Design Engineering, Imperial College London
###### *Esther Delpor, Samantha Foong, Sena Kutluay, Hannah Knight*


## Introduction to the Dataset 

Every year the UK Department of Education (DfE) publishes a dataset comparing schools and colleges exam performance across England (School and College Performance Tables). The dataset consists of school census information (e.g. pupil demographics), examination entries and results collected from exam boards and local authorities as well as data from the previous academic year. In order to protect the confidentiality of individual students, only school-level results are published. 

There are two core measures of GCSE grades, the Attainment 8 and Progress 8 scores. The Attainment 8 score is calculated by measuring a student’s performance across 8 GCSE-level qualifications (Key Stage 4), including Maths and English (which are both double weighted), from the DfE’s approved list of qualifications. Progress 8 Scores aim to capture the progress a student has made from the end of primary school to the end of secondary school compared to other students of similar prior attainment at the end of Key Stage 2.  

The full dataset consists of many more accompanying variables (e.g. break down of Attainment 8 scores for different demographics and subjects) than the ones we have chosen for this analysis, but we have selected the main demographics available. 

## Data Preparation

Our selected dataset for this analysis consists of the unique reference numbers (URNs) of 3577 schools across England, and 18 unique attributes including, postcode, school type, religious character, admissions policy, gender of entry, total number of pupils on roll, number of boys and girls at the end of key stage four, Key Stage 2 average points score of the cohort, percentage of pupils with special educational needs (SEN), total school expenditure, and Attainment 8 and Progress 8 scores in 2017-18. The dependent variables measured are the Attainment 8 and Progress 8 scores in 2018-19. 

Independent schools and new schools were filtered out as these schools could not be properly compared to other schools due to lack of data. Schools without any Key Stage 4 entries and schools with suppressed data (5 or fewer pupils taking GCSEs) were also excluded. Schools without any published Attainment 8 score for 2018-19 were also excluded as they lacked the main dependent measure.

## Information About Our Data Sources

**Our Datasets** 

https://www.gov.uk/government/statistics/key-stage-4-performance-2019-revised

**How the performance of schools and colleges is measured** 

https://www.gov.uk/government/publications/understanding-school-and-college-performance-measures/understanding-school-and-college-performance-measures?fbclid=IwAR2_qkyMT1z0IRu7ORT_bTgGEQkn-LBjeEccZGQOpsfJVtwbXATBsiUulOg


