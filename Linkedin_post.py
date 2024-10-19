import requests, json, openai
import re

class LinkedinAutomate:
    def __init__(self, access_token, openai_api,prompt):
        self.prompt=prompt
        self.access_token = access_token
        self.openai_api = openai_api
        self.python_group_list = []
        self.headers = {
            'Authorization': f'Bearer {self.access_token}'
        }

    def get_generated_content(self):
        """Use OpenAI to generate content (title and description) for LinkedIn posts."""
        client = openai.OpenAI(api_key=self.openai_api)
        DEFAULT_SYSTEM_PROMPT = '''You are a content generator. Create a captivating title and short description for a LinkedIn post.
        Format response as:
        Title: Your title
        Description: Your description (max 2-3 sentences)'''
        
        response = client.chat.completions.create(
            model= "gpt-4o-mini",
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": self.prompt}
            ]
        )
        
        # Error handling for OpenAI response
        if response.choices:
            generated_content = response.choices[0].message.content
            title_match = re.search(r"Title:(.+)", generated_content)
            description_match = re.search(r"Description:(.+)", generated_content)
            title = title_match.group(1).strip() if title_match else "Untitled"
            description = description_match.group(1).strip() if description_match else "No description provided."
            return title, description
        else:
            return "Untitled", "No description provided."

    def common_api_call_part(self, feed_type="feed", group_id=None):
        """Prepare the payload for LinkedIn post."""
        title, description = self.get_generated_content()
        
        payload_dict = {
            "author": f"urn:li:person:{self.user_id}",
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {
                        "text": description
                    },
                    "shareMediaCategory": "NONE",
                    "media": []
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC" if feed_type == "feed" else "CONTAINER"
            }
        }
        
        if feed_type == "group" and group_id:
            payload_dict["containerEntity"] = f"urn:li:group:{group_id}"
        
        return json.dumps(payload_dict)
    
    def get_user_id(self):
        url = "https://api.linkedin.com/v2/userinfo"
        response = requests.request("GET", url, headers=self.headers)
        jsonData = json.loads(response.text)
        return jsonData["sub"]
    
    def feed_post(self):
        url = "https://api.linkedin.com/v2/ugcPosts"
        payload = self.common_api_call_part()
        return requests.request("POST", url, headers=self.headers, data=payload)

    def group_post(self, group_id):
        url = "https://api.linkedin.com/v2/ugcPosts"
        payload = self.common_api_call_part(feed_type="group", group_id=group_id)
        return requests.request("POST", url, headers=self.headers, data=payload)

    def main_func(self):
        self.user_id = self.get_user_id()  # Add this line to get user ID

        feed_post = self.feed_post()
        print(feed_post)
        for group_id in self.python_group_list:
            print(group_id)
            group_post = self.group_post(group_id)
            print(group_post)
        return str(feed_post)
