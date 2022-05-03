import boto3
from pathlib import Path
from reproducible_ephys_functions import save_data_path
from one.api import ONE

one = ONE(base_url='https://openalyx.internationalbrainlab.org/')
data_path = save_data_path(figure='figure9_10')
S3_DATA_PATH = 'paper_reproducible_ephys/mtnn'


def download_aws(folder):
    repo_json = one.alyx.rest('data-repository', 'read', id='aws_cortexlab')['json']
    bucket_name = repo_json['bucket_name']
    session_keys = {
        'aws_access_key_id': repo_json.get('Access key ID', None),
        'aws_secret_access_key': repo_json.get('Secret access key', None)
    }
    session = boto3.Session(**session_keys)
    s3 = session.resource('s3')
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


def download_trained():
    if data_path.joinpath('trained_models').exists():
        return
    print('downloading trained_models')
    download_aws('trained_models')
    # TODO long filename problem
