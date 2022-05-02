import boto3
from pathlib import Path
from reproducible_ephys_functions import save_data_path

data_path = save_data_path(figure='figure9_10')
s3 = boto3.resource('s3')
S3_BUCKET_IBL = 'ibl-brain-wide-map-public'
S3_DATA_PATH = 'paper_reproducible_ephys/mtnn'


def download_aws(folder):
    bucket = s3.Bucket(S3_BUCKET_IBL)
    for obj in bucket.objects.filter(Prefix=f'{S3_DATA_PATH}/{folder}'):
        download_path = data_path.joinpath(Path(obj.key).relative_to(S3_DATA_PATH))
        download_path.parent.mkdir(exist_ok=True, parents=True)
        bucket.download_file(obj.key, str(download_path))


def download_priors():
    if data_path.joinpath('priors').exists():
        return
    print('downloading prior data')
    download_aws('priors')


def download_glm_hmm():
    if data_path.joinpath('glm_hmm').exists():
        return
    print('downloading glm_hmm data')
    download_aws('glm_hmm')


def download_trained():
    if data_path.joinpath('trained_models').exists():
        return
    print('downloading trained_models')
    download_aws('trained_models')
