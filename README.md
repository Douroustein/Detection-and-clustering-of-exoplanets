# Exoplanets detection and clustering

## Problem to be solved

### What are the main project idea and goals

Exoplanets (planets outside the Sun's Solar System) detection and clustering

### What story behind

Human kind is interestested to know, since ages, if any extraterrestrial life exist, and more recently if there is, at least, any other earth-like planet, on which we are able to leave (this interest is probably mainly due to overpopulation, lack of resources, global warming, pollution... on earth nowadays)

As a starting point, Kepler spacecraft's mission, started in 2009, is to list possible exoplanet candidates, based on planetary transits measurements :

    |When a planet passes in front of its parent star, as seen from our solar system, it blocks a small fraction of the light
    |from that star; this is known as a transit. Searching for transits of distant Earths is like looking for the drop in
    |brightness when a moth flies across a searchlight.
    |Source : NASA Exoplanet Archive : https://exoplanetarchive.ipac.caltech.edu/docs/KeplerMission.html

### What is the main motivation behind the project

    |Kepler has a fixed field of view (FOV) against the sky... Kepler's field of view covers 115 square degrees, around 0.25 
    |percent of the sky, or "about two scoops of the Big Dipper". Thus, it would require around 400 Kepler-like telescopes to
    |cover the whole sky.
    |Source : Wikipedia : https://en.wikipedia.org/wiki/Kepler_(spacecraft)
    
Based on the above fast calculation, putting in place a ML model will Simplify and fasten exoplanets detection, especially if available data will potentially increase, and will also help to cluster them into groups of similarities

## The data set

### What is the size and format of the data

For the project need, I will use 2 datasets :

##### - For detection (is the KOI is an exoplanet or a false positive)

The dataset contains 140 features (documented in https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html), 7140 data points for the train/validation/test set (Confirmed as an exoplanet or as a false positive) and 2424 datapoints for the prediction set (candidates that might be confirmed or not later on). It's a cumulative table with a summary of available data for each KOI observation :

    |The cumulative KOI table gathers information from the individual KOI activity tables that describe the current results 
    |of different searches of the Kepler light curves. The intent of the cumulative table is to provide the most accurate 
    |dispositions and stellar and planetary information for all KOIs in one place. All the information in this table has 
    |provenance in other KOI activity tables.
    |Source: https://exoplanetarchive.ipac.caltech.edu/docs/PurposeOfKOITable.html#cumulative

From litterature point of view, the dataset (or at least a part of it only contains 50 variables), has been submitted in Kaggle (https://www.kaggle.com/nasa/kepler-exoplanet-search-results). Some interesting Exploratory data analysis have been made on this dataset but no modeling or clustering :

- For the 1st analysis (https://www.kaggle.com/ezietsman/kepler-dataset-exploratory-analysis/log), the author is going through some features. The most interesting is about the Magnitude, where we clearly see a cut-off around value 16. "This is likely the limit at which the noise becomes too much to detect planetary candidates" as mentioned in the analysis. This suggest that the use of sensors with a higher efficiency might detect more planetary candidates.

- For the 2nd analysis (https://www.kaggle.com/marceltorretta/kepler-exoplanets-search-results-eda/comments), the author focused more on the data structure that are very useful:

    - NA’s: Main contributor is Kepler_name, then koi_score. For NA’s on the numerical values, he suggest that it might result from the acquisition type
    - Disposition : koi_disposition vs koi_pdisposition. 
    - False positive flags : highly correlated with false positive but not limited to. Analysis is based on koi_pdisposition.
    - Few numerical values distributions. The analysis is made based on koi_pdisposition.
        -	The koi_score is highly correlated to the koi_disposition
        -	Some of the histograms display a bimodal character: koi_period, koi_depth, koi_prad and koi_model_snr.
        -	On all bimodal histograms, the second modes always have a proportionally higher composition of ‘FALSE POSITIVES’.
        -	koi_teq and koi_insol also display some clear distinction on disposition distribution.
    - False positive flags vs numerical values: The trivial thresholds hypothesis seems implausible.
    - Correlation matrix: Confirms no correlation between False positive flags and numerical values. Other correlations exist
    - PCA applied : 8 component contain 95% of the variance on the data

- In the 3rd (https://www.kaggle.com/residentmario/automated-feature-selection-with-sklearn/comments) and 4th (https://www.kaggle.com/residentmario/ml-visualization-with-yellowbrick-1/notebook) analysis, the author was more trying to apply respectively an automated methodology and using features visualization with yellowbrick library for feature selection purpose in an easier and faster way

Another repository from Kaggle for exoplanet detection to be metioned is https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data/home, but this one is based on flux timeseries dataset, that is used to define "Threshold-Crossing Event (TCE) Information" (https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html), included in our dataset.

Conclusions we can make from these analyses (mostly from the 2nd one), and that can be used in this project:
- Main contributors for NA's have to be removed from the features list
- Analyse data acquisition type impact on NA's on numerical values as suggested by the author
- koi_pdisposition is not reliable as it changes quite often and depend on the latest analysis.
- For analysis based on koi_pdisposition, an update based on koi_disposition might be interesting
- koi_score to be challenged with detection model
- PCA to be checked for the bigger set of features


##### - For clustering (unsupervised learning)

The confirmed exoplanet dataset contains 355 features (documented in https://exoplanetarchive.ipac.caltech.edu/docs/API_exoplanet_columns.html) and 3890 data points and gathers confirmed exoplanets from different sources (55 facilities) from 1989 to 2019 (30 years) where Kepler is clearly the highest contributor.

From litterature point of view, we can see 2 repositories in Kaggle.

The 1st one (https://www.kaggle.com/mrisdal/open-exoplanet-catalogue), only uses 25 variables from the set with different names (and 3584 confirmed planets as it's from 2 years ago, some additionnal planets have been confirmed since then). Here is the list of analysis made on this dataset and some of the conclusions we can make :
- Plot planets position, which is nice to see but doesn’t bring too much information (except that universe is huge, and that there is still to much to discover)
- Find stars and planets similar to ours and possibly habitable planets by comparing exoplanets features to our solar system ones (mainly earth, Jupiter and sun). This is an interesting use case, but limiting to a comparison taking a very few samples as reference (https://www.kaggle.com/ptaswist/hospitable-planets-exploration, https://www.kaggle.com/darrellulm/earth-like-planet-explore, https://www.kaggle.com/atrexler/where-should-we-go-v2, https://www.kaggle.com/jpdeleon/exoplanets-galore, https://www.kaggle.com/etakla/destination-unknown, https://www.kaggle.com/etakla/exoploring-the-exoplanets, https://www.kaggle.com/dkleefisch/are-sun-and-earth-unique)
- Analysis of the year and method of discovery. These analyses are nicely highlighting the huge impact of Kepler’s mission using transit discovery method on the amount of discovered exoplanets. In addition to have more of them, Kepler’s mission was able to identify smaller exoplanets. Of course it doesn’t mean that other discovery method are obsolete as transit discovery method is not very adapted to detect exoplanets with a high orbital period (https://www.kaggle.com/mrisdal/space-is-the-place, https://www.kaggle.com/henergy555/exoplanet-detection, https://www.kaggle.com/pranavcoder/exoplanet-analysis, https://www.kaggle.com/donyoe/discoveries-by-method, https://www.kaggle.com/ajroyer/space-is-a-place, https://www.kaggle.com/jpdeleon/exoplanets-galore, https://www.kaggle.com/albertneutzner/analysing-methods-for-exoplanet-discovery, https://www.kaggle.com/tedrand/looking-at-the-stars, https://www.kaggle.com/inareous/discovering-planets-since-1781)
- Have a look at the stars (https://www.kaggle.com/crandelmaker/are-those-hot-stars, https://www.kaggle.com/tedrand/analyzing-discovered-planets)
- Check if there is correlation between variables (https://www.kaggle.com/playfulsynapse/visual-exploration-of-the-planets, https://www.kaggle.com/nelnour123/hidden-reationships, https://www.kaggle.com/adhok93/analysis-of-oec-data)
- Study missing values:
    - This one is proposing a methodology to replace missing values (https://www.kaggle.com/dkleefisch/imputing-missings-by-well-fitted-models)
    - This one is interestingly rising the question of correlation between missing values and discovery method (https://www.kaggle.com/sankethmopuru/some-facts-about-exoplanets)
- PCA analysis: Interesting results about correlation between discovery method and period/semi major axis features (https://www.kaggle.com/benjams/rogue-g-values-and-other-space-oddities)
- Exoplanet clustering. In this analysis, the author used the MeanShift algorithm with only 'PlanetaryMassJpt' and 'SemiMajorAxisAU' features as an input (https://www.kaggle.com/krisspe/exoplanet-clustering)

The 2nd repository (https://www.kaggle.com/eduardowoj/exoplanets-database) uses 98 variables and 3733 confirmed planets. Both authors made features value visualization
- https://www.kaggle.com/pcharambira/exoplanet-analysis)
- In this second one, the author also tried to establish linear regression  between some features (https://www.kaggle.com/justinaut/exploring-and-visualizing-exoplanet-data)

Conclusions we can make from all these analyses, and that can be used in this project:
- Missing values vs used detection method to be checked
- A more complete PCA analysis with the extended features is needed
- How clustering would be considereing the outputs of the complete PCA analysis

### How data is extracted and managed

The data is available on NASA Exoplanet Archive, Data tab (https://exoplanetarchive.ipac.caltech.edu/docs/data.html) in interactive tables. From these tables, we can extract selected (or all) data in csv format.

### What are the main challenges

- Selecting features : find out which features are usable (given missing values, wrong values) and accurate (in term of significance) for detection
- Having a high accuracy : by testing different models and tuning their parameters
- Does clustering provide a significant classification: By testing various clustering algorithms and interpret the results
