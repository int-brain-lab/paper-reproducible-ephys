import boto3
from pathlib import Path
from reproducible_ephys_functions import save_data_path
from one.api import ONE
from one.remote.aws import get_s3_from_alyx

one = ONE(base_url='https://openalyx.internationalbrainlab.org/')
data_path = save_data_path(figure='fig_mtnn')
S3_DATA_PATH = 'paper_reproducible_ephys/mtnn_Q2_2024'


def download_aws(folder):

    s3, bucket_name = get_s3_from_alyx(one.alyx)
    bucket = s3.Bucket(bucket_name)

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


def download_lp():
    if data_path.joinpath('lpks').exists():
        return
    print('downloading lightening pose data')
    download_aws('lpks')


def download_data():
    print('downloading data')
    download_aws('trained_models')
    download_aws('mtnn_data')
    download_aws('glm_data')
    download_aws('simulated_data')
