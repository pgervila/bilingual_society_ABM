
import pytest
from unittest.mock import patch
import numpy as np
import os, sys
from imp import reload
#sys.path.append("/Users/PG/Paolo/python_repos/language_proj/lang_model_simple/")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import agent
import model
reload(agent)
reload(model)
from agent import Child, Adolescent, Adult, Worker, Teacher
from model import LanguageModel

@pytest.fixture(scope="module")
def model():
    return LanguageModel(500, num_clusters=3) 

#@pytest.mark.parametrize()
def test_group_students_per_year(model):
    # check that num_employees_per_school is the same as num_courses_per_school
    num_employees_per_school = [len(model.geo.clusters_info[x]['schools'][y].info['employees'])
                                for x in range(model.geo.num_clusters) 
                                for y in range(len(model.geo.clusters_info[x]['schools']))]

    num_courses_per_school = [len(model.geo.clusters_info[x]['schools'][y].grouped_studs)
                              for x in range(model.geo.num_clusters) 
                              for y in range(len(model.geo.clusters_info[x]['schools']))]
    
    assert num_employees_per_school == num_courses_per_school

def test_school_set_up_and_update(model):
    i = np.random.randint(0, 3)
    school = model.geo.clusters_info[i]['schools'][0]
    #school.update_courses()
    # check that there are students and teacher for each existing course in school
    for (k, ags) in school.grouped_studs.items():
        assert ags['students']
        assert ags['teacher']
    # check students older than 18 are out of school
    while 18 not in school.grouped_studs:
        for st in school.info['students']:
            st.info['age'] += 36
            if isinstance(st, Child) and st.info['age'] >= st.age_high * st.model.steps_per_year:
                st.evolve(Adolescent)
        school.update_courses()
    studs = school.grouped_studs[18]['students']
    for st in school.info['students']:
        st.info['age'] += 36
    school.update_courses()
    i_max = np.argmax(model.geo.cluster_sizes)
    univ = model.geo.clusters_info[i_max]['university']
    # check some students are correctly enrolled in univ
    # check all working links with school are erased after completing education
    for stud in studs:
        assert stud not in school.info['students']
        assert stud.loc_info['school'] == None
        if 'university' in stud.loc_info:
            assert stud.loc_info['university'][0] == univ
        else:
            assert 'course_key' not in stud.loc_info

            
def test_univ_set_up_and_update(model):
    # get agents from schools to send them to univ
    schools = [school for cl_info in model.geo.clusters_info.values() 
               for school in cl_info['schools']]
    for school in schools:
        # all_studs = [x for ags in school.grouped_studs.values()
        #              for x in ags['students']]
        while school.info['age_range'][1] not in school.grouped_studs:
            for st in list(school.info['students']):
                st.info['age'] += 36
                if isinstance(st, Child) and st.info['age'] >= st.age_high * model.steps_per_year:
                    st.evolve(Adolescent)
            school.update_courses()
        for st in list(school.info['students']):
            st.info['age'] += 36
        school.update_courses()
    i_max = np.argmax(model.geo.cluster_sizes)
    univ = model.geo.clusters_info[i_max]['university']
    for fac in univ.faculties.values():
        for (k, ags) in fac.grouped_studs.items():
            print(fac, k, ags)
            assert ags['students']
            assert ags['teacher']
    for fac in univ.faculties.values():
        for st in list(fac.info['students']):
            st.info['age'] += 36
        fac.update_courses()
        for (k, ags) in fac.grouped_studs.items():
            assert ags['students']
            assert ags['teacher']
    # pick one faculty with students to check exit method
    fac = [fac for fac in univ.faculties.values() 
           if fac.info['students']][0]
    while univ.info['age_range'][1] not in fac.grouped_studs:
        for st in list(fac.info['students']):
            st.info['age'] += 36
        fac.update_courses()
    jm_studs = fac.grouped_studs[univ.info['age_range'][1]]['students']
    fac.update_courses()
    for stud in jm_studs:
        assert stud not in fac.info['students']
        assert stud not in fac.univ.info['students']
        assert stud.loc_info['university'] == None
        assert 'course_key' not in stud.loc_info

def test_hire_teachers(model):
    i = np.random.randint(0,3)
    school = model.geo.clusters_info[i]['schools'][0]
    rand_courses = np.random.choice(list(school.grouped_studs), 
                                    size=2, replace=False)
    for rand_course in rand_courses:
        school.grouped_studs[rand_course]['teacher'] = None
    school.hire_teachers(rand_courses)
    for rand_course in rand_courses:
        t = school.grouped_studs[rand_course]['teacher']
        assert t.loc_info['job'] == school
        assert t.loc_info['course_key'] == rand_course
    
def test_teachers_course_swap(model):
    for i in range(model.num_clusters):
        school = model.geo.clusters_info[i]['schools'][0]
        sorted_keys = sorted(list(school.grouped_studs.keys()))
        sorted_keys =  sorted_keys if not len(sorted_keys) % 2 else sorted_keys[:-1]
        ts_bf = [school.grouped_studs[k]['teacher'] for k in sorted_keys]
        school.swap_teachers_courses()
        ts_after = [school.grouped_studs[k]['teacher'] for k in sorted_keys]
        for pair_bf, pair_aft in zip(zip(*[iter(ts_bf)] * 2), zip(*[iter(ts_after)] * 2)):
            assert pair_bf == pair_aft[::-1]
                  
def test_assign_new_stud_to_course(model):
    home = model.geo.clusters_info[0]['homes'][0]
    new_student = Adolescent(model, model.num_people + 1,
                             1, 'M', home=home, age = 13 * 36 + 5 )
    school = model.geo.clusters_info[0]['schools'][0]
    school.assign_stud(new_student)
    new_stud_course = int(new_student.info['age'] / model.steps_per_year)
    assert new_student in school.grouped_studs[new_stud_course]['students']
    assert new_student.loc_info['course_key'] == new_stud_course
    
    
    