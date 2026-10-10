from googleapiclient.discovery import build


API_KEY = "AIzaSyDbSXZZoq0yE3EOHHWq7hLt7cf160Nh0DM"



def get_video_id(url):
    """
    Extract YouTube video ID from a normal YouTube URL.
    """

    if "v=" in url:
        return url.split("v=")[1].split("&")[0]

    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]

    return url


def get_comments(video_url, max_comments=100):

    video_id = get_video_id(video_url)

    youtube = build(
        "youtube",
        "v3",
        developerKey=API_KEY
    )

    comments = []
    next_page_token = None

    while len(comments) < max_comments:

        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            pageToken=next_page_token,
            textFormat="plainText",
            order="time"
        )

        response = request.execute()

        for item in response.get("items", []):

            comment = item["snippet"]["topLevelComment"]["snippet"]

            comments.append({
                "author": comment.get("authorDisplayName", ""),
                "comment": comment.get("textDisplay", ""),
                "published_at": comment.get("publishedAt", ""),
                "likes": comment.get("likeCount", 0)
            })

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

    return comments[:max_comments]