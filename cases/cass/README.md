Usage:

Create /global/homes/m/mpowell/.cdsapirc with your ADS credentials:


url: https://cds.climate.copernicus.eu/api/v2
key: <your-cds-uid>:<your-cds-api-key>
url_ads: https://ads.atmosphere.copernicus.eu/api/v2
key_ads: <your-ads-uid>:<your-ads-api-key>
Create the output directory:


mkdir -p /pscratch/sd/m/mpowell/LS2D_CAMS
Run it:


python download_cams.py
ADS queues requests asynchronously — if it exits saying "request not finished", just re-run it; the .pickle files track the pending requests.