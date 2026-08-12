import requests
from bs4 import BeautifulSoup
import json
import time
import re

def build_polyu_glossary_index():
    # Expanded category sources incorporating all requested PolyU glossary sections
    category_sources = {
        "Buildings & Facilities": [
            "https://www.polyu.edu.hk/web/glossary/en/terms_relating_to_structure_and_organization_post/buildings_facilities/index.html"
        ],
        "Faculties and Academic Departments": [
            "https://www.polyu.edu.hk/web/glossary/en/terms_relating_to_structure_and_organization_post/faculties_and_academic_departments/index.html"
        ],
        "Terms Relating to Research, Consultancy and Academic Collaboration": [
            "https://www.polyu.edu.hk/web/glossary/en/terms_relating_to_research_consultancy_and_academic_collaboration/index.html"
        ],
        "Academic Awards": [
            "https://www.polyu.edu.hk/web/glossary/en/terms_relating_to_students_and_programmes/academic_awards/index.html"
        ],
        "Boards, Committees and Related Bodies": [
            "https://www.polyu.edu.hk/web/glossary/en/terms_relating_to_structure_and_organization_post/boards_committees_and_related_bodies/index.html"
        ],
        "College of Professional and Continuing Education": [
            "https://www.polyu.edu.hk/web/glossary/en/terms_relating_to_structure_and_organization_post/college_of_professional_and_continuing_education/index.html"
        ],
        "Other Departments/Offices/Units": [
            "https://www.polyu.edu.hk/web/glossary/en/terms_relating_to_structure_and_organization_post/other_departments_offices_units/index.html"
        ],
        "Post titles": [
            "https://www.polyu.edu.hk/web/glossary/en/terms_relating_to_structure_and_organization_post/post_titles/index.html"
        ],
        "Academic Rules": [
            "https://www.polyu.edu.hk/web/glossary/en/terms_relating_to_students_and_programmes/student_admission_registration_examinations/index.html"
            
        ],
        
        "Course Syllabi":[
            "https://www.polyu.edu.hk/ise/study/information-for-current-students/programme-related-info/subject-syllabi/"
        ]    
    }
    
    term_index = {}
    print("Starting comprehensive PolyU glossary indexing across all requested categories...")

    for category, urls in category_sources.items():
        for url in urls:
            print(f"Scraping [{category}] from: {url}")
            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, "html.parser")
                    rows = soup.find_all("tr")
                    
                    count = 0
                    for row in rows:
                        cols = row.find_all("td")
                        if len(cols) >= 2:
                            cell_texts = [col.text.strip() for col in cols if col.text.strip()]
                            
                            english_name = ""
                            chinese_name = ""
                            abbreviation = ""
                            
                            for text in cell_texts:
                                if re.search(r'[\u4e00-\u9fff]', text):
                                    chinese_name = text
                                elif text.isupper() and len(text) <= 8 and not " " in text:
                                    abbreviation = text
                                else:
                                    if not english_name:
                                    # Fallback if multiple text segments exist
                                        english_name = text
                                        
                            if not english_name and len(cell_texts) > 0:
                                english_name = cell_texts[0]

                            primary_key = abbreviation if abbreviation else english_name
                            
                            if primary_key:
                                term_index[primary_key] = {
                                    "english": english_name,
                                    "chinese": chinese_name,
                                    "abbreviation": abbreviation,
                                    "category": category
                                }
                                count += 1
                                
                    print(f"-> Extracted {count} entries for {category}.")
            except Exception as e:
                print(f"-> Error parsing {url}: {e}")
            time.sleep(1)

    # High-Priority Manual Overrides including precise WIE Chinese naming
    manual_overrides = {
        "ISE": {
            "english": "Department of Industrial and Systems Engineering", 
            "chinese": "工業及系統工程學系", 
            "abbreviation": "ISE", 
            "category": "Faculties and Academic Departments"
        },
        "WIE": {
            "english": "Work-Integrated Education", 
            "chinese": "校企協作教育/實習", 
            "abbreviation": "WIE", 
            "category": "Student Affairs"
        },
        "GUR": {
            "english": "General University Requirements", 
            "chinese": "學院通識要求", 
            "abbreviation": "GUR", 
            "category": "Academic Rules"
        }
    }
    term_index.update(manual_overrides)

    with open("term_index.json", "w", encoding="utf-8") as f:
        json.dump(term_index, f, indent=4, ensure_ascii=False)
        
    print(f"\nSuccessfully generated term_index.json with {len(term_index)} fully indexed items across all target definitions!")

if __name__ == "__main__":
    build_polyu_glossary_index()