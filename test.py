import google.generativeai as genai
genai.configure(api_key="AIzaSyAhi5t6zay1LeuMMlsxdGut7ZtUuSt2cvs")
m = genai.GenerativeModel("gemini-2.5-flash")
r = m.generate_content("Say hi!")
print(r.text)