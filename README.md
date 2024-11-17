# consumption_agg_calc

This code currently assumes you have the [Malawi microdata](https://microdata.worldbank.org/index.php/catalog/3818) .dta files in a folder called "MWI_2019_IHS-V_v06_M_STATA" in the top level directory.

Places we can improve:
- We can use the exact date of the survey to calculate the recall period for each good. Currently we just assume a mean month length.
- We can make sure we're in the exact same units as their consumption aggregate. We might not be in  