#UTS Microbiota
***
Microbiota residing on the urban transit systems (UTSs) can be shared by travelers, and they have niche-specific assemblage. However, it remains unclear how the assembly processes are influenced by city characteristics, rendering city-specific and microbial-aware urban planning challenging.

In this study, we analyzed 3359 UTS microbial samples collected from 16 cities around the world. We found the stochastic process dominated in all UTS microbiota assemblages, with the explanation rate (R^2^) of the neutral community model (NCM) higher than 0.7. However, city characteristics predominantly drove the assembly, largely responsible for the variation in the stochasticity ratio. Furthermore, by utilizing an artificial intelligence model, we quantified the ability of UTS microbes in discriminating between cities and found that the ability was strongly affected by city characteristics. For example, although the NCM R^2^ of the New York City UTS microbiota was 0.831, the accuracy of the microbial-based city characteristic classifier was higher than 0.9. This is the first study to demonstrate the effects of city characteristics on the UTS microbiota assemblage, paving the way for city-specific and microbial-aware applications.

The assigned taxonomic table and metadata could be downloaded from [MetaSUB](https://pngb.io/metasub-2021). We present the results and code of our analyses to [GitHub](https://github.com/XiongGZ/UTS_Microbiota) to reproduce the analysis or learn from it.

##Directory and file Illustration
***
###01. Assemblage
Explore the assemblage of UTS microbiota.

**tNST.R**
- Calculate the taxonomic normalized stochasticity ratio ([tNST](https://github.com/DaliangNing/NST)) of UTS samples using metadata and a taxonomic table as input.

**pairwise.csv**
- The output of tNST.R, includes stochasticity ratio (ST), normalized stochasticity ratio (NST), modified stochasticity ratio (MST), standard effect size (SES), and modified Raup-Crick index (RC)

###02.Source_tracking
Explore the association within the microbial communities on different New York City UTS surfaces.

**source_tracking.R**
-  Assign microbiota on one of NYC UTS surfaces as sink, and calculate the contribution of the microbiota on the other 13 NYC UTS surfaces using fast expectation-maximization microbial source tracking ([FEAST](https://github.com/cozygene/FEAST)). Metadata (containing environment, sampleid, and sink/source information) and taxonomic tables are needed as input.

**metadata**
- The metadata tables which has three columns (i.e., 'Env', 'SourceSink', 'id'). The first column is a description of the sampled environment (e.g., human gut), and the second column indicates if this sample is a source or a sink (can take the value 'Source' or 'Sink'). The third column is the Sink-Source id. For more reference format details refer to [FEAST](https://github.com/cozygene/FEAST).

**result**
- The result of NYC UTS surface microbiota source track using FEAST. 

###03.Specificity_quantify
Quantify the ability of UTS microbes that have top 100 square deviations in discriminating between cities designed as city specificity.

**vars_top100.csv**
- The UTS microbes with top 100 square deviations in relative abundance per surface type (bench, door, handrail, and kiosk).

**`transferRF.py`**
- Construct three kinds of random forest models, i.e., base model (BM), independent model (IM), and transfer model (TM), and evaluate the AUROC of constructed models.

**data_configuration**
- Contain three kinds of files, i.e. Query*, Source, and Transfer* which include the sampleid and city information of the query dataset, source dataset, and transfer dataset respectively. For more details on data configuration refer to the paper.

**result**
- The result of quantifying UTS microbe city specificity

###04.Characteristic_predict
Confirm the association between the UTS microbes and city characteristics based on which city characteristics can be accurately predicted by using city-specific microbes.

**`predict.py`**
- Construct the BM, IM, and TM using city-specific or non-city-specific UTS microbes as features, and calculate the accuracy of these models.

**feature_top100.csv**
- The UTS microbes with top 100 city specificity per surface type (bench, door, handrail, and kiosk).

**data_configuration**
- Contain three kinds of files, i.e. Query*, Source, and Transfer* which include the sampleid and city information of the query dataset, source dataset, and transfer dataset respectively.

**result**
- The result of predicting city characteristics using random forest models using UTS microbes as features.

###TransferRandomForest
Implementation of the SER-STRUCT algorithm to perform transfer learning with random forest. The origin package is [TransferRandomForest](https://github.com/Luke3D/TransferRandomForest). We add the evaluation function to this package.

The algorithm is presented in this paper: 

http://ieeexplore.ieee.org/document/7592407/
