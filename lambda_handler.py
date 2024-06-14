from audio_similarity_search import get_audio, get_embeddings, compute_sim


def lambda_handler(event, context):
    s3_obj = event["Records"][0]["s3"]
    s3_bucket = s3_obj["bucket"]["name"]
    s3_key = s3_obj["object"]["key"]

    reference_audio = get_audio(s3_bucket, s3_key)
    return "downloaded_audio"
    similarity_scores = []
    objects = s3.list_objects_v2(Bucket=s3_bucket)

    for obj in objects.get("Contents", []):
        target_audio = get_audio(s3_bucket, obj["Key"])
        reference_emb = get_embeddings(reference_audio)
        target_emb = get_embeddings(target_audio)
        similarity = compute_sim(reference_emb, target_emb)
        similarity_scores.append((obj["Key"], similarity.item()))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    return {"similarity_scores": similarity_scores}

if __name__ == "__main__":
    event = {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": "sample_all_video_files"},
                    "object": {"key": "family_specific_sample.mp3"},
                }
            }
        ]
    }
    lambda_handler(event, None)
