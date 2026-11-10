"""Upload the exported models to a Cloudflare R2 bucket through its S3 API.

Usage: CLOUDFLARE_API_TOKEN=... CLOUDFLARE_ACCOUNT_ID=... python upload_r2.py <models dir> [bucket]
The S3 credentials are derived from the API token (access key = token id, secret = sha256 of the
token), which needs R2 write permission. Files already uploaded with the same size are skipped.
Weights get a one-year immutable cache (the app requests them as <file>?v=<sha256 prefix>).
The bucket also needs CORS for the site's origin and, for public access, a custom domain."""
import hashlib, json, os, sys, urllib.request
import boto3
from boto3.s3.transfer import TransferConfig

token = os.environ['CLOUDFLARE_API_TOKEN']
account = os.environ['CLOUDFLARE_ACCOUNT_ID']
req = urllib.request.Request('https://api.cloudflare.com/client/v4/user/tokens/verify', headers={'Authorization': f'Bearer {token}'})
token_id = json.load(urllib.request.urlopen(req))['result']['id']
s3 = boto3.client('s3', endpoint_url=f'https://{account}.r2.cloudflarestorage.com', aws_access_key_id=token_id,
                  aws_secret_access_key=hashlib.sha256(token.encode()).hexdigest(), region_name='auto')
src = sys.argv[1]
bucket = sys.argv[2] if len(sys.argv) > 2 else 'kronos-models'
cfg = TransferConfig(multipart_threshold=64 * 2**20, multipart_chunksize=64 * 2**20, max_concurrency=4)
existing = {o['Key']: o['Size'] for o in s3.list_objects_v2(Bucket=bucket).get('Contents', [])}
for name in sorted(os.listdir(src)):
    path = os.path.join(src, name)
    size = os.path.getsize(path)
    if name == 'manifest.json':
        extra = {'ContentType': 'application/json', 'CacheControl': 'no-cache'}
    elif name.endswith('.onnx'):
        extra = {'ContentType': 'application/octet-stream', 'CacheControl': 'public, max-age=31536000, immutable'}
    else:
        continue
    if existing.get(name) == size and name != 'manifest.json':
        print('skip', name); continue
    s3.upload_file(path, bucket, name, ExtraArgs=extra, Config=cfg)
    print('up', name, size)
