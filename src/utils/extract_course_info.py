import pandas as pd

def extract_course_info(api_response):
    df_course = pd.read_csv('src\\datasets\\course_info_trans.csv')
    
    course_ids = api_response[0]

    courses_info = []
    
    for course_id in course_ids:
        course_info = df_course[df_course['id'] == course_id]
        
        if not course_info.empty:
            info_dict = {
                'id': course_id,
                'name': course_info.iloc[0]['name_trans'],
                'about': course_info.iloc[0]['about_trans']
            }
            courses_info.append(info_dict)
        else:
            courses_info.append({'id': course_id, 'name': None, 'about': None})
    
    return courses_info
    
