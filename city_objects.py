import random
import numpy as np
from itertools import groupby
from math import ceil
import string
from scipy.spatial.distance import pdist

from agent import Adolescent, Young, YoungUniv, Adult, Teacher, TeacherUniv, Pensioner


class Home:
    def __init__(self, clust, pos):
        self.pos = pos
        self.agents_in = set()
        self.info = {'occupants': set(), 'clust': clust}

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
                curr_clust = ag.loc_info['home'].info['clust']
                ag.loc_info['home'].remove_agent(ag)
                # change cluster info reference to agent if new home is in another cluster
                if self.info['clust'] != curr_clust:
                    ag.model.geo.update_agent_clust_info(ag, curr_clust=curr_clust,
                                                         update_type='switch',
                                                         new_clust=self.info['clust'])
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
        return 'Home_{0[clust]!r}_{1.pos!r}'.format(self.info, self)


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
        """
            Method to hire all necessary suitable teacher agents for a new school instance.
            It first calls a method to group students by age to form courses, then it assigns
            a teacher to each course. Method is called ONLY at model initialization
        """
        if not self.grouped_studs:
            print('grouping', self)
            self.group_students_per_year()
        self.hire_teachers(self.grouped_studs.keys())

    def update_courses_phase_1(self, max_course_dist=3):
        """
            Method to (partially) update courses at the end of the year. It moves all students forward to
            next course and checks each new course has a corresponding teacher. It sends a percentage of
            last year students to university, rest to job market
            Args:
                * max_course_dist : integer. Maximum distance in years between courses to allow
                    teacher exchange
        """

        if not self.grouped_studs:
            self.group_students_per_year()
        else:
            # define empty set to update school groups
            updated_groups = {}

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

    def update_courses_phase_2(self):
        """
            Method to hire missing teachers after phase1-update and
            replace retired teachers
        """

        retirement_age = Teacher.age_high * self.model.steps_per_year

        for c_id, c_info in list(self.grouped_studs.items()):
            if not c_info['teacher']:
                self.hire_teachers([c_id])

        # check if teacher has to retire and replace it if needed
        for (c_id, course) in self.grouped_studs.items():
            if course['teacher'].info['age'] >= retirement_age:
                course['teacher'].evolve(Pensioner)

        #print('UPDATED ', self)

    def find_teachers(self, courses_keys):
        """
        Find teachers for the specified courses. Method looks first among available
        courseless teachers in educational centers from current cluster, then from other clusters.
        If not all places can be covered, method hires ordinary employees from closest to farthest
        clusters
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

        # loop over clusters from closest to farthest from school, to hire NON teachers
        num_missing_teachers = num_needed_teachers - len(hired_teachers)
        new_teachers = self.get_employees_from_companies(num_missing_teachers)
        hired_teachers.extend(new_teachers)
        hired_teachers = set(hired_teachers[:num_needed_teachers])
        return hired_teachers

    def check_teacher_old_job(self, teacher):
        """
            Method to check if an agent already had a job before
            being hired as a teacher. After identifying the job type,
            method removes agent from former job
            Args:
                * teacher: Teacher class instance
            Output:
                * agent removal from former job, if any
        """
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

    def hire_teachers(self, courses_keys, move_home=True):
        """
            Hire teachers for the specified courses
            Args:
                * courses_keys: list of integer(s). Identifies courses for which teachers are missing
                    through years of age of its students
                * move_home: force hired teacher to move home when accepting new position (if
                    teacher's home and school are not in the same cluster)
        """
        # TODO : sort employees by lang competence from lowest to highest
        # TODO : set conditions for hiring according to students age. Higher age, higher requirements

        new_teachers = self.find_teachers(courses_keys)

        # block hired teachers from being hired by other schools (avoid modifying set while looping)
        for teacher in new_teachers:
            teacher.blocked = True
        # Assign course keys and educ center to teachers
        # make copy of zipped iterables to avoid 'dict changed size during iter' exception
        # if a child of a new teacher is enrolled in a non existing course when moving, exception would occur
        # because a new course would be added to courses_keys while iterating

        teacher_type = Teacher if type(self) == School else TeacherUniv
        for (ck, teacher) in list(zip(courses_keys, new_teachers)):
            # turn hired agent into appropriate teacher_type instance if it's not yet one
            if not isinstance(teacher, teacher_type):
                teacher = teacher.evolve(teacher_type, ret_output=True)
            self.assign_teacher(teacher, ck, move_home=move_home)
        # delete blocked attribute
        for teacher in new_teachers:
            del teacher.blocked

    def get_free_staff_from_cluster(self):
        """ This method will be implemented in subclasses """
        pass

    def get_free_staff_from_other_clusters(self):
        """ This method will be implemented in subclasses """
        pass

    def get_employees_from_companies(self, num_teachers):
        """
            Method to get a list of agents with an ordinary job to be hired as teachers
            in school. Method looks from closest to farthest cluster relative to school
            Args:
                * num_teachers: integer. Number of teachers requested
            Output:
                * list of agents
        """
        center_clust = self.info['clust']
        # loop over clusters from closest to farthest from school, to hire employees as teachers
        for ix in self.model.geo.clusters_info[center_clust]['closest_clusters']:
            # list cluster teacher candidates.
            # TODO : hire teachers based on fact they have UNIV education !!!
            new_teachers = [ag for ag in self.model.geo.clusters_info[ix]['agents']
                            if ag.info['language'] in self.info['lang_policy'] and
                            ag.info['age'] > (self.info['min_age_teacher'] * self.model.steps_per_year) and
                            not isinstance(ag, (Teacher, TeacherUniv, Pensioner))]
            # keep only one agent per marriage to avoid excessive recursion in agent moving
            candidates, consorts = [], []
            for ag in new_teachers:
                if ag not in consorts:
                    candidates.append(ag)
                    if ag.info['married']:
                        consorts.append(ag.get_family_relative('consort'))
            new_teachers = candidates
            # filter out agents whose consort is a Teacher
            new_teachers = [ag for ag in new_teachers
                            if ag.info['married'] and type(ag.get_family_relative('consort')) != Teacher]
            # filter out agents that are currently being hired in a given school by using ag 'blocked' attribute
            # it is possible that because of cascading in school jobs when families move after a consort is hired,->
            # a school hires an agent that is in the process of being hired by another school (in the set queue)

            new_teachers[:] = [ag for ag in new_teachers if not hasattr(ag, 'blocked')]

            # shuffle agents to add randomness
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
            Output:
                * course_key
        """

        # First check which center, if any, the student comes from, and remove links to it
        try:
            old_educ_center, course_key = student.get_school_and_course()
        except (KeyError, TypeError):
            old_educ_center = None
        if old_educ_center and old_educ_center is not self:
            old_educ_center.remove_student(student)

        # check if we are in initializat mode and if school has already been initialized
        if self.model.init_mode and not self.grouped_studs:
            student.loc_info['school'] = [self, None]
            self.info['students'].add(student)
        else:
            # if no course_key from former school, create one to register into new school
            if not course_key:
                course_key = int(student.info['age'] / self.model.steps_per_year)
            # Now assign student to new educ_center and corresponding course
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
        """ Method will be implemented in subclasses """
        pass

    def remove_student_from_course(self, student, educ_center, replace=None,
                                   grown_agent=None, upd_course=False):
        """
            Remove students from their course, and optionally replace them with grown_agents
            Args:
                * student: agent instance
                * educ_center: string. It is either 'school' or 'university'
                * replace: boolean. If True, grown_agent must be specified
                * grown_agent: agent instance. It must be specified if replace value is True
                * upd_course: boolean. False if removal does not involve student exit towards university
                    or job market
        """
        course_key = student.loc_info[educ_center][1]
        # after updating courses every year, leaving students (for Univ or JobMarket) are no longer
        # amongst students in course set and thus cannot be removed from it.
        # Remove from course only if student has not left school
        if not upd_course:
            self[course_key]['students'].remove(student)
        student.loc_info[educ_center][1] = None
        if replace and educ_center == 'school' and not isinstance(student, Adolescent):
            self[course_key]['students'].add(grown_agent)
            grown_agent.loc_info[educ_center][1] = course_key

        # check if there are students left in course once student has left
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
        # assign school and course to teacher
        teacher.loc_info['job'] = [self, course_key]
        # assign teacher to school if not yet belonging to it
        if teacher not in self.info['employees']:
            self.info['employees'].add(teacher)
        if move_home:
            # move to a new home if current home is not in same cluster of school
            teacher_clust = teacher['clust']
            school_clust = self.info['clust']
            if teacher_clust != school_clust:
                teacher.move_to_new_home(marriage=False)

    def get_free_staff_from_cluster(self):
        """ Method to get all free teachers from schools
            in school's cluster """

        school_clust = self.info['clust']
        cluster_candidates = [t for school in self.model.geo.clusters_info[school_clust]['schools']
                              for t in school.info['employees']
                              if not t.loc_info['job'][1] and
                              t.info['language'] in self.info['lang_policy']]

        cluster_candidates[:] = [t for t in cluster_candidates if not hasattr(t, 'blocked')]

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
            clust_free_staff[:] = [t for t in clust_free_staff if not hasattr(t, 'blocked')]
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
        # remove student from course if he/she has a course_key assigned
        course_key = student.get_school_and_course()[1]
        if course_key:
            self.remove_student_from_course(student, 'school', replace=replace,
                                            grown_agent=grown_agent, upd_course=upd_course)
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
            teacher_clust = teacher['clust']
            school_clust = self.info['clust']
            # move only if current home is not in the same cluster as job
            if teacher_clust != school_clust:
                teacher.move_to_new_home(marriage=False)

    def get_free_staff_from_cluster(self):
        """
            Method to get all courseless teachers from faculties
            in self faculty cluster
        """
        free_staff = [t for t in self.univ.info['employees']
                      if not t.loc_info['job'][1]]
        free_staff[:] = [t for t in free_staff if not hasattr(t, 'blocked')]
        return free_staff

    def get_free_staff_from_other_clusters(self, num_teachers):
        """
            Method to get all courseless teachers from faculties
            in clusters other than self faculty's one
            Args:
                * num_teachers: integer. Number of teachers requested
        """

        fac_clust = self.info['clust']
        other_clusters_free_staff = []
        for clust in self.model.geo.clusters_info[fac_clust]['closest_clusters'][1:]:
            if 'university' in self.model.geo.clusters_info[clust]:
                univ = self.model.geo.clusters_info[clust]['university']
                clust_free_staff = [t for t in univ.info['employees']
                                    if not t.loc_info['job'][1] and
                                    t.info['language'] in univ.info['lang_policy']]
                clust_free_staff[:] = [t for t in clust_free_staff if not hasattr(t, 'blocked')]
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
            # TODO : change following line !!!!
            self.info['lang_policy'] = [0, 1]
        self.faculties = {key: Faculty(key, self, model) for key in string.ascii_letters[:5]}

    def __getitem__(self, fac_key):
        return getattr(self, 'faculties')[fac_key]

    def __repr__(self):
        return 'University_{0[clust]!r}_{1.pos!r}'.format(self.info, self)


class Job:
    """ class that defines a Job object.
        Args:
            * model: object instance. Instance of LanguageModel class the Job instance refers to
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
        self.pos = pos
        self.num_places = num_places
        # TODO : define network of companies (customers) where nodes are companies
        self.info = {'employees': set(), 'lang_policy': lang_policy,
                     'skill_level': skill_level, 'clust': clust}
        self.agents_in = set()
        # set lang policy
        if not lang_policy:
            self.set_lang_policy()

    def set_lang_policy(self, min_pct=0.1):
        """ Computes langs distribution in cluster the job instance belongs to,
            and sets value for job lang policy accordingly. In order to be taken
            into account, monolinguals need to make > min_pct % cluster population
            Args:
                * min_pct: float. minimum percentage for monolinguals to be taken into account.
                Defaults to 0.1
            Output:
                * sets value of self.info['lang_policy]
        """
        lang_distrib = self.model.geo.get_lang_distrib_per_clust(self.info['clust'])
        lang_policy = np.where(lang_distrib > min_pct)[0]
        if 1 not in lang_policy and lang_policy.size < 2:
            self.info['lang_policy'] = np.insert(lang_policy, np.searchsorted(lang_policy, 1), 1)
        elif 1 not in lang_policy and lang_policy.size == 2:
            self.info['lang_policy'] = np.array([1])
        elif 1 in lang_policy and lang_policy.size > 2:
            self.info['lang_policy'] = np.array([1])
        else:
            self.info['lang_policy'] = lang_policy

    def check_cand_conds(self, agent, keep_cluster=False, job_steps_years=3):
        """
            Method to check that family constraints and current job seniority
            are compatible with job hiring
            Args:
                * agent: instance.
                * job_steps_years: integer. Number of years of seniority in current job before being allowed to
                    move to a new job in another cluster
            Output:
                * True if checking is satisfactory, None otherwise
        """
        # define minimum number of job steps to be allowed to move
        min_num_job_steps = (self.model.steps_per_year * job_steps_years
                             if not keep_cluster else self.model.steps_per_year)
        # avoid hiring of teachers
        if isinstance(agent, Young) and not isinstance(agent, (Teacher, Pensioner)):
            # check agent is either currently unemployed or employed for enough time
            employm_conds = not agent.loc_info['job'] or agent.info['job_steps'] > min_num_job_steps
            if employm_conds:
                if keep_cluster:
                    return True
                else:
                    consort = agent.get_family_relative('consort')
                    if consort:
                        consort_is_teacher = isinstance(consort, Teacher)
                        if not consort_is_teacher:
                            try:
                                consort_employm_conds = (not consort.loc_info['job'] or
                                                         consort.info['job_steps'] > min_num_job_steps)
                                if consort_employm_conds:
                                    return True
                            except KeyError:
                                return True

                    else:
                        return True

    def look_for_employee(self, excluded_ag=None):
        """
            Look for employee that meets requirements: lang policy, currently unemployed and with partner
            able to move
            Args:
                * excluded_ag: agent excluded from search
                * job_steps_years: integer. Minimum number of years in a job before agent can be hired
                    again
        """

        # look for suitable agents in any cluster
        # TODO: limit number of iterations per step and shuffle agents !!!!
        for ag in set(self.model.schedule.agents).difference(set([excluded_ag])):
            keep_cluster = True if ag['clust'] == self.info['clust'] else False
            if self.check_cand_conds(ag, keep_cluster=keep_cluster) and not hasattr(ag, 'blocked'):
                self.hire_employee(ag)
                break

    def hire_employee(self, agent, move_home=True):
        """
            Method to remove agent from former employment (if any) and hire it
            into a new one. Method checks that job's language policy is compatible
            with agent language knowledge. If hiring is unsuccessful, agent reacts
            to the linguistic exclusion
            Args:
                * agent: agent instance. Defines agent that will be hired
                * move_home: boolean. True if agent is allowed to move home
        """
        # First prevent agent from being hired by other companies
        # during hiring cascade after removal from current company
        agent.blocked = True

        # Check if agent has to be removed from old job
        try:
            old_job = agent.loc_info['job']
            if old_job and old_job is not self:
                old_job.remove_employee(agent)
        except KeyError:
            pass

        # hire agent
        if agent.info['language'] in self.info['lang_policy']:
            self.num_places -= 1
            self.info['employees'].add(agent)
            # assign job to agent
            agent.loc_info['job'] = self
            # reset job seniority steps counter
            agent.info['job_steps'] = 0

            # move agent to new home closer to job if necessary (and requested)
            if move_home:
                agent_clust = agent['clust']
                job_clust = self.info['clust']
                if agent_clust != job_clust:
                    agent.move_to_new_home(marriage=False)

        # if hiring is unsuccessful, we know it is because of lang reasons
        if not agent.loc_info['job']:
            lang = 'L2' if agent.info['language'] == 0 else 'L1'
            agent.react_to_lang_exclusion(lang)

        # free agent from temporary hiring block
        del agent.blocked


    def remove_employee(self, agent, replace=None, new_agent=None):
        self.num_places += 1
        # remove all agent's links to job
        self.info['employees'].remove(agent)
        agent.loc_info['job'] = None
        if agent in self.agents_in:
            self.agents_in.remove(agent)
        # Either replace agent by new_agent or hire a new random one
        if replace:
            self.info['employees'].add(new_agent)
            new_agent.loc_info['job'] = self
        else:
            del agent.info['job_steps']
            if not self.model.init_mode:
                self.look_for_employee(excluded_ag=agent)

    def __repr__(self):
        return 'Job_{0[clust]!r}_{1.pos!r}'.format(self.info, self)


class MeetingPoint:
    pass


class Store:
    pass

class BookStore(Store):
    pass