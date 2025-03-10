# Cyclone Track Forecasting
Forecasting the track of cyclones using deep learning from METEOSAT satellite images over the 57E campaign from 2007 to 2016.

## **Satellite Image Download**
The raw images were downloaded from [EUMETSAT](https://data.eumetsat.int/product/EO:EUM:DAT:0081) and further processed to generate the training dataset. The images have three channels [(page 11)](https://user.eumetsat.int/s3/eup-strapi-media/pdf_mviri_fcdr_atbd_75cac1f577.pdf):
    - IR 108 (InfraRed at 11500 nm)
    - VIS 6 (VISible at 700 nm)
    - WV 73 (WaterVapour 6400 nm)

14,543 images were downloaded to create the dataset.


## **Dataset Construction**
The dataset is created through the following steps:

### Step 1 : Cyclone Position

The positions are obtained from the [IMD Best Track](https://rsmcnewdelhi.imd.gov.in/report.php?internal_menu=MzM=) dataset as an excel sheet. We also obtain `estimated_central_pressure`, `max_sustained surface_wind`, `pressure_drop` and `grade` too.

### Step 2. Wind Velocity Bands

Wind velocity vectors are calculated by computing the optical flow between two subsequent images, and downsampling it from 1280x1280 to 128x128. Then, the average velocity is computed in 0.5 degree bands from the centre of the cyclone, for the IR and WV channels. VIS is ignored due to diurnal variations.

### Step 3. Date Encoding

The date in the year, time of day and phase of moon are encoded as sine waves. This is to encourage the model to ensure that it captures the seasonal effects.

### Step 4. Bringing it together

All the columns from the above steps are merged together to form the CSV file [dataset.csv](./dataset.csv)

### Step 5. Combinations

For each cyclone, all valid nC4 pairs are chosen to create the final dataset mapping of inputs (any three prior timestamps in order) to output (Position of fourth timestamp).


## **Models**
We experimented with the following models:

<ADD REGRESSION MODELS>