import random
import numpy as np
from itertools import groupby
from math import ceil
import string
from scipy.spatial.distance import pdist

from agent import Adolescent, Young, YoungUniv, Adult, Teacher, TeacherUniv, Pensioner


class Home:
    def __init__(self, clust, pos):
        self.clust = clust
        self.pos = pos
        self.agents_in = set()
        self.info = {'occupants': set()}

    def assign_to_agent(self, agents):
        """ Assign the current home instance to each agent
            Args:
                * agents: agent instance or list of agent instances
        """
        # convert agents to iterable if only a single agent in input
        if not isinstance(agents, list):
            agents = [agents]
        for ag in agents:
            # check if agent already has a home
            if ag.loc_info['home']:
                curr_clust = ag.loc_info['home'].clust
                ag.loc_info['home'].remove_agent(ag)
                # change cluster info reference to agent if new home is in another cluster
                if self.clust != curr_clust:
                    ag.model.geo.update_agent_clust_info(ag, curr_clust=curr_clust,
                                                         update_type='switch', new_clust=self.clust)
            # assign new home
            ag.loc_info['home'] = self
            self.info['occupants'].add(ag)
            self.agents_in.add(ag)

    def remove_agent(self, agent, replace=False, grown_agent=None):
        """ Remove agent from home occupants. If requested,
            replace agent with a specific new_agent
            Args:
                * agent: class instance. Agent to be removed
                * replace: boolean. If True, agent will be replaced after removal
                * grown_agent: class instance. Agent that will replace former agent in home occupants
        """

        self.info['occupants'].remove(agent)
        agent.loc_info['home'] = None
        if replace:
            self.info['occupants'].add(grown_agent)
            grown_agent.loc_info['home'] = self
        if agent in self.agents_in:
            self.agents_in.remove(agent)

    def __repr__(self):
        return 'Home_{0.clust!r}_{0.pos!r}'.format(self)


class EducationCenter:
    """ Common methods to all educational centers such as Schools, Faculties """

    def __init__(self, model, pos, clust, num_places, age_range,
                 lang_policy, min_age_teacher):
        self.model = model
        self.pos = pos
        self.agents_in = set()
        self.info = {'employees': set(), 'students': set(),
                     'lang_policy': lang_policy, 'clust': clust,
                     'age_range': age_range, 'num_places': num_places,
                     'min_age_teacher': min_age_teacher}
        self.grouped_studs = dict()

    def group_students_per_year(self):
        """ group school students by age in order to create different courses """
        if not self.grouped_studs:
            list_studs = list(self.info['students'])
            if list_studs:
                studs_sorted = sorted(list_studs, key=lambda x: int(x.info['age'] / self.model.steps_per_year))
                self.grouped_studs = dict([(c_key, {'students': set(gr_studs)})
                                           for c_key, gr_studs
                                           in groupby(studs_sorted,
                                                      lambda x: int(x.info['age'] / self.model.steps_per_year))
                                           if c_key <= self.info['age_range'][1]])

    def set_up_courses(self):
        """ Method to hire all necessary suitable teacher agents for a school instance.
            It calls method to group students by age to form courses, then it assigns
            a teacher to each course
        """
        if not self.grouped_studs:
            self.group_students_per_year()
        self.hire_teachers(self.grouped_studs.keys(), move_home=False)

    def update_courses_phase_1(self, max_course_dist=3):
        """
            Method to update courses at the end of the year. It moves all students forward to
            next course and checks each new course has a corresponding teacher. It sends a percentage of
            last year students to university, rest to job market
            Args:
                * max_course_dist : integer. Maximum distance in years between courses to allow
                    teacher exchange
        """

        if not self.grouped_studs:
            self.group_students_per_year()
        else:
            print('STARTING UPDATE OF ', self)
            # define empty set to update school groups
            updated_groups = {}
            # define variable to store students from last year course( if there is such course )
            exit_students = None
            # define Teacher retirement age in years
            retirement_age = Teacher.age_high * self.model.steps_per_year
            for (c_id, course) in self.grouped_studs.items():
                # If course_id is smaller than maximum allowed age in school, rearrange
                if c_id < self.info['age_range'][1]:
                    # move students forward to next course and delete students from current course
                    updated_groups[c_id + 1] = {'students': course['students']}
                    course['students'] = None
                    # check if c_id + 1 was already part of last year courses
                    if c_id + 1 not in self.grouped_studs:
                        # set missing teacher for new course
                        updated_groups[c_id + 1]['teacher'] = None
                    else:
                        if self.grouped_studs[c_id + 1]['teacher']:
                            updated_groups[c_id + 1]['teacher'] = self.grouped_studs[c_id + 1]['teacher']
                        else:
                            updated_groups[c_id + 1]['teacher'] = None
                else:
                    # store students that will move out of school
                    exit_students = course['students']
                    self.exit_studs(exit_students)

            # define set of teachers that are left with an empty course
            jobless_teachers_keys = {key for key in self.grouped_studs
                                     if key not in updated_groups and self[key]['teacher']}

            # define set of missing teachers in updated_groups
            missing_teachers_keys = {key for key in updated_groups
                                     if not updated_groups[key]['teacher']}

            # define reallocating pairs dict {jobless_key: missing_key}
            # to reassign school jobless teachers to courses with students but without teacher
            pairs = {}
            for jlt_key in list(jobless_teachers_keys):
                for mt_key in list(missing_teachers_keys):
                    if abs(jlt_key - mt_key) <= max_course_dist and mt_key not in pairs.values():
                        pairs.update({jlt_key: mt_key})
            for (jlt_key, mt_key) in pairs.items():
                updated_groups[mt_key]['teacher'] = self[jlt_key]['teacher']
                updated_groups[mt_key]['teacher'].loc_info['job'][1] = mt_key
                # update sets after reallocation
                jobless_teachers_keys.remove(jlt_key)
                missing_teachers_keys.remove(mt_key)

            # check if there are still jobless teachers left in school
            # remove reference to course in their job spec
            # TODO : for jobless teachers, find another occupation. Use them when needed
            for jlt_key in jobless_teachers_keys:
                self[jlt_key]['teacher'].loc_info['job'][1] = None

            # assign updated groups to grouped_studs class attribute
            self.grouped_studs = updated_groups

            # assign course id to each student
            educ_center = 'school' if isinstance(self, School) else 'university'
            for (c_id, course) in self.grouped_studs.items():
                for st in course['students']:
                    st.loc_info[educ_center][1] = c_id
            # assign info needed later after all school rearrangements are done
            #self.info['update'] = {'missing_teachers_keys': missing_teachers_keys}

    def update_courses_phase_2(self):
        """ Method to hire missing teachers after phase1-update and
            replace retiring teachers"""

        retirement_age = Teacher.age_high * self.model.steps_per_year
        # missing_teachers_keys = self.info['update']['missing_teachers_keys']
        # exit_students = self.info['update']['exit_students']

        for c_id, c_info in self.grouped_studs.items():
            if not c_info['teacher']:
                self.hire_teachers([c_id])

        # # check if there are still missing teachers for some courses and hire them
        # if missing_teachers_keys:
        #     self.hire_teachers(missing_teachers_keys)

        # check if teacher has to retire and replace it if needed
        for (c_id, course) in self.grouped_studs.items():
            if course['teacher'].info['age'] >= retirement_age:
                course['teacher'].evolve(Pensioner)

        print('UPDATED ', self)

    def find_teachers(self, courses_keys):
        """
        Find teachers for the specified courses. Method looks first among available
        courseless teachers at educat_center. If not sufficient, it looks further among
        other educ_centers clusters in other clusters
            Args:
                * courses_keys: list of integer(s). Identifies courses for which teachers are missing
                    through age of its students
                * ret_output: boolean. True when output needs to be returned
        """

        num_needed_teachers = len(set([c_id for c_id in courses_keys if c_id]))
        hired_teachers = []

        # first check among teachers employed at educ_center but without course_key assigned
        free_staff_from_cluster = self.get_free_staff_from_cluster()
        hired_teachers.extend(free_staff_from_cluster)
        if len(hired_teachers) >= num_needed_teachers:
            hired_teachers = set(hired_teachers[:num_needed_teachers])
            return hired_teachers

        # loop over clusters from closest to farthest from educ_center to hire other teachers without course
        num_missing_teachers = num_needed_teachers - len(hired_teachers)
        free_staff_from_other_clust = self.get_free_staff_from_other_clusters(num_missing_teachers)
        hired_teachers.extend(free_staff_from_other_clust)
        if len(set(hired_teachers)) >= num_needed_teachers:
            hired_teachers = set(hired_teachers[:num_needed_teachers])
            return hired_teachers

        # loop over clusters from closest to farthest from school, to hire non teachers
        num_missing_teachers = num_needed_teachers - len(hired_teachers)
        new_teachers = self.get_employees_from_companies(num_missing_teachers)
        hired_teachers.extend(new_teachers)
        hired_teachers = set(hired_teachers[:num_needed_teachers])
        return hired_teachers

    def check_teacher_old_job(self, teacher):
        try:
            old_job = teacher.loc_info['job']
            if isinstance(old_job, list):
                if len(old_job) == 2:
                    old_job = old_job[0]
                else:
                    old_job = old_job[0][old_job[2]]
            if old_job and old_job is not self:
                old_job.remove_employee(teacher)
        except KeyError:
            pass

    def hire_teachers(self, keys, move_home=True):
        """ This method will be implemented in subclasses """
        pass

    def get_free_staff_from_cluster(self):
        """ This method will be implemented in subclasses """
        pass

    def get_free_staff_from_other_clusters(self):
        """ This method will be implemented in subclasses """
        pass

    def get_employees_from_companies(self, num_teachers):
        """ Args:
                * num_teachers: integer. Number of teachers requested"""
        center_clust = self.info['clust']
        # loop over clusters from closest to farthest from school, to hire employees as teachers
        for ix in self.model.geo.clusters_info[center_clust]['closest_clusters']:
            # list cluster teacher candidates. Shuffle them to add randomness
            # TODO : hire teachers based on fact they have UNIV education !!!
            new_teachers = [ag for ag in self.model.geo.clusters_info[ix]['agents']
                           if ag.info['language'] in self.info['lang_policy'] and
                           ag.info['age'] > (self.info['min_age_teacher'] * self.model.steps_per_year) and
                           not isinstance(ag, (Teacher, TeacherUniv, Pensioner))]
            random.shuffle(new_teachers)
            if len(new_teachers) >= num_teachers:
                return new_teachers
        return new_teachers

    def exit_studs(self, studs):
        """ Method will be customized in subclasses """
        pass

    def assign_student(self, student, course_key=None, hire_t=True):
        """
            Method that assigns student to educational center( school or university)
            and corresponding course.
            It creates new course in center if it does not already exist
            It checks for old school if any and removes student from it
            Args:
                * student: agent instance.
                * course_key: integer. Specify course_key
                * hire_t: boolean. If True, a teacher will be hired if the assigned course did not exist.
                    Defaults to True
        """

        # First checks center student comes from, if any, and remove links to it
        try:
            old_educ_center, course_key = student.get_school_and_course()
        except KeyError:
            old_educ_center = None
        if old_educ_center and old_educ_center is not self:
            old_educ_center.remove_student(student)

        # Now assign student to new educ_center and corresponding course
        if not course_key:
            course_key = int(student.info['age'] / self.model.steps_per_year)
        # assign student if course already exists, otherwise create course
        if course_key in self.grouped_studs:
            self[course_key]['students'].add(student)
        else:
            self.grouped_studs[course_key] = {'students': {student}, 'teacher': None}
            if hire_t:
                self.hire_teachers([course_key])
        self.info['students'].add(student)
        return course_key

    def remove_student(self, student):
        """ Method will be implemented in subclasses"""
        pass

    def remove_student_from_course(self, student, educ_center, replace=None,
                                   grown_agent=None, upd_course=False):
        """
            Remove students from their course, and optionally replace them with grown_agents
            Args:
                * student: agent instance
                * educ_center: string. It is either 'school' or 'university'
                * replace:boolean. If True, grown_agent must be specified
                * grown_agent: agent instance
                * upd_course: boolean. False if removal does not involve exit towards university or job market
        """
        course_key = student.loc_info[educ_center][1]
        # after updating courses, leaving student is no longer amongst students in course
        # thus remove from course only if not leaving school
        if not upd_course:
            self[course_key]['students'].remove(student)
        student.loc_info[educ_center][1] = None
        if replace and educ_center == 'school' and not isinstance(student, Adolescent):
            self[course_key]['students'].add(grown_agent)
            grown_agent.loc_info[educ_center][1] = course_key

        # check if there are students left in course
        if not upd_course and not self[course_key]['students']:
            self.remove_course(course_key)

    def remove_teacher_from_course(self, teacher, replace=None, new_teacher=None):
        if teacher.loc_info['job'][1]:
            course_key = teacher.loc_info['job'][1]
            self[course_key]['teacher'] = None
            teacher.loc_info['job'][1] = None
            if replace:
                self[course_key]['teacher'] = new_teacher
                new_teacher.loc_info['job'][1] = course_key

    def remove_agent_in(self, agent):
        if agent in self.agents_in:
            self.agents_in.remove(agent)

    def remove_course(self, course_key):
        """ Remove school course if no students are left in it. Relocate teacher """
        if not self[course_key]['students']:
            courseless_teacher = self[course_key]['teacher']
            if courseless_teacher:
                courseless_teacher.loc_info['job'][1] = None
            del self.grouped_studs[course_key]

    def __getitem__(self, key):
        return getattr(self, 'grouped_studs')[key]


class School(EducationCenter):
    """ Class that defines a School object
        Args:
            * pos: 2-D tuple of integers. School coordinates
            * clust: integer. Cluster to which school belongs
            * num_places: integer. Total number of school places available
            * age_range: 2-D tuple of integers.
            * lang_policy: requested languages in order to work at this site
                [0, 1] -> both 0, 1 agents may work here
                [1] -> only 1 agents may work here
                [1, 2] -> both 1, 2 agents may work here
    """
    def __init__(self, model, pos, clust, num_places, age_range=(1, 18),
                 lang_policy=None, min_age_teacher=30):
        self.model = model
        self.pos = pos
        self.agents_in = set()
        self.info = {'employees': set(), 'students': set(),
                     'lang_policy': lang_policy, 'clust': clust,
                     'age_range': age_range, 'num_places': num_places,
                     'min_age_teacher': min_age_teacher}
        self.grouped_studs = dict()

    def group_students_per_year(self):

        super().group_students_per_year()

        # assign course key to students
        for c_key, ags in self.grouped_studs.items():
            for st in ags['students']:
                st.loc_info['school'][1] = c_key

    def assign_teacher(self, teacher, course_key=None, move_home=True):
        """ Method to assign school job to a teacher. It checks that all links
            to previous job are erased before assigning the new ones
            Args:
                * teacher: Teacher class instance.
                * course_key: integer. It specifies teacher's course, if any. It defaults to none
                * move_home: boolean. If True(default), it forces teacher agent to
                    move to a new home if his current home is not in the same cluster as the new school
        """

        self.check_teacher_old_job(teacher)

        if course_key:
            # assign teacher to course
            self.grouped_studs[course_key]['teacher'] = teacher
        teacher.loc_info['job'] = [self, course_key]
        if teacher not in self.info['employees']:
            self.info['employees'].add(teacher)
        if move_home:
            teacher_clust = teacher.loc_info['home'].clust
            school_clust = self.info['clust']
            if teacher_clust is not school_clust:
                teacher.move_to_new_home(marriage=False)

    def hire_teachers(self, courses_keys, move_home=True):
        """
            Hire teachers for the specified courses
            Args:
                * courses_keys: list of integer(s). Identifies courses for which teachers are missing
                    through years of age of its students
                * move_home: force hired teacher to move home when accepting new position
        """
        hired_teachers = self.find_teachers(courses_keys)
        # assign class key to teachers and add teachers to grouped studs
        # TODO : sort employees by lang competence from lowest to highest
        # TODO : set conditions for hiring according to students age. Higher age, higher requirements
        for (k, hired_t) in zip(courses_keys, hired_teachers):
            if not isinstance(hired_t, Teacher):
                # turn hired agent into Teacher
                hired_t = hired_t.evolve(Teacher, ret_output=True)
            self.assign_teacher(hired_t, k, move_home=move_home)

    def get_free_staff_from_cluster(self):
        """ Method to get all free teachers from schools
            in school's cluster """

        school_clust = self.info['clust']
        cluster_candidates = [t for school in self.model.geo.clusters_info[school_clust]['schools']
                              for t in school.info['employees']
                              if not t.loc_info['job'][1] and
                              t.info['language'] in self.info['lang_policy']]
        return cluster_candidates

    def get_free_staff_from_other_clusters(self, num_teachers):
        """
            Method to get teachers without course assigned from clusters other than current one
            Args:
                * num_teachers: integer. Number of teachers requested
        """
        school_clust = self.info['clust']
        other_clusters_free_staff = []
        for clust in self.model.geo.clusters_info[school_clust]['closest_clusters'][1:]:
            clust_free_staff = [t for sc in self.model.geo.clusters_info[clust]['schools']
                                         for t in sc.info['employees'] if not t.loc_info['job'][1]
                                         and t.info['language'] in self.info['lang_policy']]
            other_clusters_free_staff.extend(clust_free_staff)
            if len(other_clusters_free_staff) >= num_teachers:
                return other_clusters_free_staff
        return other_clusters_free_staff

    def assign_student(self, student, course_key=None, hire_t=True):

        course_key = super().assign_student(student, course_key=course_key, hire_t=hire_t)
        student.loc_info['school'] = [self, course_key]

    def exit_studs(self, studs):
        """ Method to send studs from last year to univ or job market """

        universities = [clust_info['university']
                        for clust_info in self.model.geo.clusters_info.values()
                        if 'university' in clust_info]
        # get closest university from school location
        idx_univ = np.argmin([pdist([self.pos, univ.pos])
                              for univ in universities])
        univ = universities[idx_univ]
        # approx half of last year studs to univ
        univ_stds = np.random.choice(list(studs),
                                     size=ceil(0.5 * len(studs)),
                                     replace=False)
        for st in univ_stds:
            st.evolve(YoungUniv, university=univ, upd_course=True)
        # rest of last year students to job market
        job_stds = set(studs).difference(set(univ_stds))
        for st in job_stds:
            st.evolve(Young, upd_course=True)

    def swap_teachers_courses(self):
        """ Every n years on average teachers are swapped between classes """
        # sort dict keys before iteration
        sorted_keys = sorted(list(self.grouped_studs.keys()))
        # swap teacher keys, and teachers for each pair of classes
        for (k1, k2) in zip(*[iter(sorted_keys)] * 2):
            self[k1]['teacher'].loc_info['job'][1] = k2
            self[k2]['teacher'].loc_info['job'][1] = k1
            (self[k1]['teacher'],
             self[k2]['teacher']) = (self[k2]['teacher'], self[k1]['teacher'])

    def remove_student(self, student, replace=False, grown_agent=None, upd_course=False):
        self.info['students'].remove(student)
        self.remove_agent_in(student)
        # course_key
        self.remove_student_from_course(student, 'school', replace=replace, grown_agent=grown_agent,
                                        upd_course=upd_course)
        # replace agent only if it is not an Adolescent instance
        if replace and not isinstance(student, Adolescent):
            self.info['students'].add(grown_agent)
        else:
            student.loc_info['school'] = None

    def remove_employee(self, teacher, replace=False, new_teacher=None):
        self.info['employees'].remove(teacher)
        self.remove_agent_in(teacher)
        # course_key
        course_key = teacher.loc_info['job'][1]
        if course_key:
            self.remove_teacher_from_course(teacher, replace=replace, new_teacher=new_teacher)
        if replace:
            # if replacement is because of growing agent
            self.info['employees'].add(new_teacher)
            self.agents_in.add(new_teacher)
        else:
            teacher.loc_info['job'] = None
            # teacher needs to be replaced anyway (e.g. Pensioner)
            if course_key:
                self.hire_teachers([course_key])

    def __repr__(self):
        return 'School_{0[clust]!r}_{1.pos!r}'.format(self.info, self)


class Faculty(EducationCenter):

    def __init__(self, fac_type, univ, model):
        self.model = model
        self.univ = univ
        self.pos = self.univ.pos
        self.agents_in = set()
        self.info = {'students': set(), 'employees': set(),
                     'lang_policy': univ.info['lang_policy'],
                     'age_range': univ.info['age_range'],
                     'min_age_teacher': univ.info['min_age_teacher'], 'type': fac_type,
                     'clust': univ.info['clust']}
        #self.grouped_studs = {k: defaultdict(set) for k in range(*self.info['age_range'])}
        self.grouped_studs = dict()

    def group_students_per_year(self):

        super().group_students_per_year()

        # assign course key to students
        for c_key, ags in self.grouped_studs.items():
            for st in ags['students']:
                st.loc_info['university'][2] = c_key

    def assign_teacher(self, teacher, course_key=None, move_home=True):

        self.check_teacher_old_job(teacher)

        if course_key:
            self.grouped_studs[course_key]['teacher'] = teacher
        teacher.loc_info['job'] = [self.univ, course_key, self.info['type']]
        if teacher not in self.info['employees']:
            self.info['employees'].add(teacher)
            self.univ.info['employees'].add(teacher)
        # check if moving to a new home is needed
        if move_home:
            teacher_clust = teacher.loc_info['home'].clust
            school_clust = self.info['clust']
            if teacher_clust is not school_clust:
                teacher.move_to_new_home(marriage=False)

    def hire_teachers(self, courses_keys):
        """ Hire teachers for the specified courses
            Args:
                * courses_keys: list of integer(s). Identifies courses for which teachers are missing
                    through years of age
        """
        hired_teachers = self.find_teachers(courses_keys)
        # assign class key to teachers and add teachers to grouped studs
        # TODO : sort employees by lang competence from lowest to highest
        for (k, t) in zip(courses_keys, hired_teachers):
            # turn hired agent into Teacher
            if not isinstance(t, TeacherUniv):
                t = t.evolve(TeacherUniv, ret_output=True)
            self.assign_teacher(t, k)

    def get_free_staff_from_cluster(self):
        """ Method to get all courseless teachers from faculties
            in self faculty cluster """
        free_staff = [t for t in self.univ.info['employees']
                      if not t.loc_info['job'][1]]
        return free_staff

    def get_free_staff_from_other_clusters(self, num_teachers):
        """ Method to get all courseless teachers from faculties
            in clusters other than self faculty's one
            Args:
                * num_teachers: integer. Number of teachers requested"""

        fac_clust = self.info['clust']
        other_clusters_free_staff = []
        for clust in self.model.geo.clusters_info[fac_clust]['closest_clusters'][1:]:
            if 'university' in self.model.geo.clusters_info[clust]:
                univ = self.model.geo.clusters_info[clust]['university']
                clust_free_staff = [t for t in univ.info['employees']
                                    if not t.loc_info['job'][1] and
                                    t.info['language'] in univ.info['lang_policy']]
                other_clusters_free_staff.extend(clust_free_staff)
                if len(other_clusters_free_staff) >= num_teachers:
                    return other_clusters_free_staff
        return other_clusters_free_staff

    def exit_studs(self, studs):
        for st in list(studs):
            st.evolve(Young, upd_course=True)

    def assign_student(self, student, course_key=None, hire_t=True):
        course_key = super().assign_student(student, course_key=course_key, hire_t=hire_t)
        self.univ.info['students'].add(student)
        student.loc_info['university'] = [self.univ, course_key, self.info['type']]

    def remove_student(self, student, upd_course=False):
        # remove from uni and fac
        self.univ.info['students'].remove(student)
        self.info['students'].remove(student)
        # course_key
        self.remove_student_from_course(student, 'university', upd_course=upd_course)
        self.remove_agent_in(student)
        student.loc_info['university'] = None

    def remove_employee(self, teacher, replace=False, new_teacher=None):
        self.univ.info['employees'].remove(teacher)
        self.info['employees'].remove(teacher)
        self.remove_agent_in(teacher)
        # course key
        course_key = teacher.loc_info['job'][1]
        if course_key:
            self.remove_teacher_from_course(teacher, replace=replace, new_teacher=new_teacher)
        if replace:
            self.univ.info['employees'].add(new_teacher)
            self.info['employees'].add(new_teacher)
        else:
            teacher.loc_info['job'] = None
            # teacher needs to be replaced anyway (e.g. Pensioner)
            if course_key:
                self.hire_teachers([course_key])

    def __repr__(self):
        return 'Faculty_{0[clust]!r}_{1.pos!r}_{0[type]!r}'.format(self.info, self)


class University:
    """ class that defines a University object for tertiary studies """

    def __init__(self, model, pos, clust, age_range=(19, 23),
                 lang_policy=None, min_age_teacher=35):
        self.model = model
        self.pos = pos
        self.info = {'clust': clust, 'lang_policy': lang_policy, 'students': set(),
                     'employees': set(), 'age_range': age_range, 'min_age_teacher': min_age_teacher}
        if lang_policy:
            self.info['lang_policy'] = lang_policy
        else:
            self.info['lang_policy'] = [0, 1]
        self.faculties = {key: Faculty(key, self, model) for key in string.ascii_letters[:5]}

    def __getitem__(self, fac_key):
        return getattr(self, 'faculties')[fac_key]

    def __repr__(self):
        return 'University_{0[clust]!r}_{1.pos!r}'.format(self.info, self)


class Job:
    """ class that defines a Job object.
        Args:
            * clust: integer. Index of cluster where job object is found
            * pos: 2-D integer tuple. Job location on grid
            * num_places: integer. Num of available job offers.
            * lang_policy: requested languages in order to work at this site
                0 -> only L1 (so both 0, 1 agents may work here)
                1 -> both L1 and L2 ( only 1 agents may work here )
                2 -> only L2 (so both 1, 2 agents may work here)
    """
    def __init__(self, model, clust, pos, num_places, skill_level=0, lang_policy=None):
        self.model = model
        self.clust = clust
        self.pos = pos
        self.num_places=num_places
        self.info = {'employees': set(), 'lang_policy': lang_policy,
                     'skill_level': skill_level }
        self.agents_in = set()

    def look_for_employee(self, excluded_ag=None):
        """
            Look for employee that meets requirements
            Args:
                * excluded_ag: agent excluded from search
        """

        # look for suitable agents in any cluster
        for ag in set(self.model.schedule.agents).difference(set([excluded_ag])):
            if isinstance(ag, Young) and not isinstance(ag, (Teacher, Pensioner)):
                if ag.info['language'] in self.info['lang_policy'] and not ag.loc_info['job']:
                    self.hire_employee(ag)
                    break

    def hire_employee(self, agent, move_home=True):
        try:
            old_job = agent.loc_info['job']
            if old_job and old_job is not self:
                old_job.remove_employee(agent)
        except KeyError:
            pass
        if agent.info['language'] in self.info['lang_policy']:
            self.num_places -= 1
            agent.loc_info['job'] = self
            self.info['employees'].add(agent)
            # move agent to new home closer to job if necessary (and requested)
            if move_home:
                agent_clust = agent.loc_info['home'].clust
                job_clust = self.clust
                if agent_clust is not job_clust:
                    agent.move_to_new_home(marriage=False)
    # TODO : update workers by department and send them to retirement when age reached

    def remove_employee(self, agent, replace=None, new_agent=None):
        self.num_places += 1
        self.info['employees'].remove(agent)
        agent.loc_info['job'] = None
        if agent in self.agents_in:
            self.agents_in.remove(agent)
        # Either replace agent by new_agent or hire a new random one
        if replace:
            self.info['employees'].add(new_agent)
            new_agent.loc_info['job'] = self
        else:
            self.look_for_employee(excluded_ag=agent)

    def __repr__(self):
        return 'Job_{0.clust!r}_{0.pos!r}'.format(self)


class MeetingPoint:
    pass


class Store:
    pass

class BookStore(Store):
    pass