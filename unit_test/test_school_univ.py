
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
from agent import BaseAgent, Child, Adolescent, YoungUniv, Young, Adult, Worker, Teacher
from model import LanguageModel

@pytest.fixture(scope="module")
def model():
    return LanguageModel(500, num_clusters=3)

@pytest.fixture(scope="module")
def univ(model):
    ix_sorted_clusts_by_size = np.argsort(model.geo.cluster_sizes)[::-1]
    for ix_clust in ix_sorted_clusts_by_size:
        try:
            univ = model.geo.clusters_info[ix_clust]['university']
            break
        except KeyError:
            continue
    return univ


def test_group_students_per_year(model):
    # check that num_employees_per_school is the same as num_courses_per_school
    num_employees_per_school = [len(model.geo.clusters_info[x]['schools'][y].info['employees'])
                                for x in range(model.geo.num_clusters) 
                                for y in range(len(model.geo.clusters_info[x]['schools']))]

    num_courses_per_school = [len(model.geo.clusters_info[x]['schools'][y].grouped_studs)
                              for x in range(model.geo.num_clusters) 
                              for y in range(len(model.geo.clusters_info[x]['schools']))]
    
    assert num_employees_per_school == num_courses_per_school


def test_school_set_up_and_update(model, univ):
    i = np.random.randint(0, 3)
    school = model.geo.clusters_info[i]['schools'][0]
    # check that there are students and teacher for each existing course in school
    for (k, ags) in school.grouped_studs.items():
        assert ags['students']
        assert ags['teacher']
        assert type(ags['teacher']) == Teacher
    # check students older than 18 are out of school
    while 18 not in school.grouped_studs:
        for st in list(school.info['students'])[:]:
            st.info['age'] += 36
            if isinstance(st, Child) and st.info['age'] >= st.age_high * st.model.steps_per_year:
                st.evolve(Adolescent)

        school.update_courses_phase_1()
        school.update_courses_phase_2()
        for k, c_info in school.grouped_studs.items():
            assert c_info['teacher'].loc_info['job'][1] == k
        num_course_teachers = len([t for t in school.info['employees'] 
                                   if t.loc_info['job'][1]])
        num_courses = len(school.grouped_studs)
        assert num_course_teachers == num_courses
    studs = school.grouped_studs[18]['students'].copy()
    for st in list(school.info['students'])[:]:
        st.info['age'] += 36
        if isinstance(st, Child) and st.info['age'] >= st.age_high * st.model.steps_per_year:
            st.evolve(Adolescent)
    school.update_courses_phase_1()
    school.update_courses_phase_2()
    for (k, ags) in school.grouped_studs.items():
        assert ags['students']
        assert ags['teacher']
        assert type(ags['teacher']) == Teacher
    # check some students are correctly enrolled in univ
    # check all working links with school are erased after completing school education

    for stud in studs:
        assert stud not in school.info['students']
        assert 'school' not in stud.loc_info
        if 'university' in stud.loc_info:
            assert stud.loc_info['university'][0] == univ
                
    # test teacher retirement
    rand_teacher = np.random.choice([t for t in school.info['employees'] 
                                     if t.loc_info['job'][1]])
    while (rand_teacher.info['age'] / 36 ) <= 65:
        rand_teacher.info['age'] += 36
    course_key = rand_teacher.loc_info['job'][1]
    school.update_courses_phase_1()
    school.update_courses_phase_2()
    if course_key in school.grouped_studs:
        if school.grouped_studs[course_key]['students']:
            assert school.grouped_studs[course_key]['teacher']
        else:
            assert not school.grouped_studs[course_key]['teacher']
    
    # test students exit
    for _ in range(3):
        for st in list(school.info['students'])[:]:
            st.info['age'] += 36
            if isinstance(st, Child) and st.info['age'] >= st.age_high * st.model.steps_per_year:
                st.evolve(Adolescent)
        school.update_courses_phase_1()
        school.update_courses_phase_2()
        for (k, ags) in school.grouped_studs.items():
            assert ags['students']
            assert ags['teacher']
            assert type(ags['teacher']) == Teacher
            assert ags['teacher'].loc_info['job'][1] == k


def test_univ_set_up_and_update(model, univ):
    # get agents from schools to send them to univ
    schools = [school for cl_info in model.geo.clusters_info.values() 
               for school in cl_info['schools']]
    for school in schools:
        # all_studs = [x for ags in school.grouped_studs.values()
        #              for x in ags['students']]
        while school.info['age_range'][1] not in school.grouped_studs:
            for st in list(school.info['students'])[:]:
                st.info['age'] += 36
                if isinstance(st, Child) and st.info['age'] >= st.age_high * model.steps_per_year:
                    st.evolve(Adolescent)
            school.update_courses_phase_1()
            school.update_courses_phase_2()
        for st in list(school.info['students'])[:]:
            st.info['age'] += 36
            if isinstance(st, Child) and st.info['age'] >= st.age_high * model.steps_per_year:
                st.evolve(Adolescent)
        school.update_courses_phase_1()
        school.update_courses_phase_2()

    for fac in univ.faculties.values():
        fac.update_courses_phase_2()
        for (k, ags) in fac.grouped_studs.items():
            # print(fac, k, ags)
            assert ags['students']
            assert ags['teacher']
    for fac in univ.faculties.values():
        for st in list(fac.info['students'])[:]:
            st.info['age'] += 36
            if isinstance(st, YoungUniv) and st.info['age'] >= st.age_high * model.steps_per_year:
                st.evolve(Young)
        fac.update_courses_phase_1()
        fac.update_courses_phase_2()
        for (k, ags) in fac.grouped_studs.items():
            assert ags['students']
            assert ags['teacher']
    # pick one faculty with students to check exit method
    fac = [fac for fac in univ.faculties.values() 
           if fac.info['students']][0]
    while univ.info['age_range'][1] not in fac.grouped_studs:
        for st in list(fac.info['students'])[:]:
            st.info['age'] += 36
        fac.update_courses_phase_1()
        fac.update_courses_phase_2()
    # store students that will move to job market
    jm_studs = fac.grouped_studs[univ.info['age_range'][1]]['students']
    fac.update_courses_phase_1()
    fac.update_courses_phase_2()
    for stud in jm_studs:
        assert stud not in fac.info['students']
        assert stud not in fac.univ.info['students']
        assert 'university' not in stud.loc_info


def test_hire_teachers(model):
    i = np.random.randint(0, 3)
    school = model.geo.clusters_info[i]['schools'][0]
    if len(school.grouped_studs) >= 2:
        rand_courses = np.random.choice(list(school.grouped_studs), 
                                        size=2, replace=False)
        for rand_course in rand_courses:
            school.grouped_studs[rand_course]['teacher'] = None
        school.hire_teachers(rand_courses)
        for rand_course in rand_courses:
            t = school.grouped_studs[rand_course]['teacher']
            assert t.loc_info['job'][0] == school
            assert t.loc_info['job'][1] == rand_course
    else:
        pass

    # test moving to a new home


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
    school.assign_student(new_student)
    new_stud_course = int(new_student.info['age'] / model.steps_per_year)
    assert new_student in school.grouped_studs[new_stud_course]['students']
    assert new_student.loc_info['school'][1] == new_stud_course

def test_teacher_death(model):
    school = model.geo.clusters_info[0]['schools'][0]
    ck = list(school.grouped_studs.keys())[0]
    teacher = school[ck]['teacher']
    with patch('agent.BaseAgent.random_death') as mock_rand_death:
        mock_rand_death.return_value = True
        teacher.random_death()
        assert school[ck]['teacher'] != teacher
        assert school[ck]['teacher']