import random
import numpy as np
from itertools import groupby
from math import ceil
import string
from scipy.spatial.distance import pdist

from agent import Young, YoungUniv, Adult, Teacher, TeacherUniv, Pensioner


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
                ag.loc_info['home'].info['occupants'].remove(ag)
                # change cluster info reference to agent if new home is in another cluster
                if self.clust != curr_clust:
                    ag.model.geo.update_agent_clust_info(ag, curr_clust=curr_clust,
                                                         update_type='switch', new_clust=self.clust)
            # assign new home
            ag.loc_info['home'] = self
            self.info['occupants'].add(ag)
            self.agents_in.add(ag)


        # TODO : remove agents from old homes and clusters, add to new clusters

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
                # assign course key to students
                for c_key, ags in self.grouped_studs.items():
                    for st in ags['students']:
                        st.loc_info['course_key'] = c_key

    def find_teachers(self, courses_keys, ret_output=False):
        """
        Hire teachers for the specified courses
            Args:
                * courses_keys: list of integer(s). Identifies courses for which teachers are missing
                    through years of age
                * ret_output: boolean. True when output needs to be returned
        """
        # TODO : when teacher dies or retires =>
        # TODO a signal needs to be sent to school so that it can update => Call 'look_for_teachers'
        num_needed_teachers = len(courses_keys)
        school_clust_ix = self.info['clust']
        hired_teachers = []
        # loop over clusters from closest to farthest from school
        for ix in self.model.geo.clusters_info[school_clust_ix]['closest_clusters']:
            # list cluster teacher candidates. Shuffle them to add randomness
            # TODO : hire teachers based on fact they have UNIV education !!!
            clust_cands = [ag for ag in self.model.geo.clusters_info[ix]['agents']
                           if ag.info['language'] in self.info['lang_policy'] and
                           ag.info['age'] > (self.info['min_age_teacher'] * self.model.steps_per_year) and
                           not isinstance(ag, (Teacher, TeacherUniv))]
            random.shuffle(clust_cands)
            hired_teachers.extend(clust_cands)
            if len(hired_teachers) >= num_needed_teachers:
                # hire all teachers and assign school to them before stopping loop
                hired_teachers = set(hired_teachers[:num_needed_teachers])
                break
            else:
                # continue looking for missing teachers in next cluster
                continue
        if ret_output:
            return hired_teachers

    def hire_teachers(self, keys):
        """ This method will be fully implemented in subclasses"""
        pass

    def exit_studs(self, studs):
        """ Method will be customized in subclasses """
        pass

    def set_up_courses(self):
        """ Method to hire all necessary suitable teacher agents for a school instance.
            It calls method to group students by age to form courses, then it assigns
            a teacher to each course
        """
        if not self.grouped_studs:
            self.group_students_per_year()
        self.hire_teachers(self.grouped_studs.keys())

    def assign_stud(self, student):
        course_key = int(student.info['age'] / self.model.steps_per_year)
        if course_key in self.grouped_studs:
            # if course already exists
            self.grouped_studs[course_key]['students'].add(student)
        else:
            self.grouped_studs[course_key] = {'students': {student}}
            self.hire_teachers([course_key])
        self.info['students'].add(student)
        student.loc_info['course_key'] = course_key

    def update_courses(self, max_course_dist=3):
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
            # define empty set to update school groups
            updated_groups = {}
            # define set with keys for next year courses that will have students but still have no teacher
            missing_teachers_keys = set()
            for (c_id, course) in self.grouped_studs.items():
                # If course_id is smaller than maximum allowed age in school, rearrange
                if c_id < self.info['age_range'][1]:
                    # move students forward to next class and delete students from current class
                    updated_groups[c_id + 1] = {'students': course['students']}
                    course['students'] = None
                    # check if c_id + 1 was already part of last year courses
                    if c_id + 1 not in self.grouped_studs:
                        # if it's a new course, add course id to missing teacher keys
                        missing_teachers_keys.add(c_id + 1)
                    else:
                        # keep teacher
                        updated_groups[c_id + 1].update({'teacher': self.grouped_studs[c_id + 1]['teacher']})
                    # check if teacher has to retire
                    if course['teacher'].info['age'] >= course['teacher'].age_high * self.model.steps_per_year:
                        course['teacher'].evolve(Pensioner)
                        missing_teachers_keys.add(c_id)
                else:
                    self.exit_studs(course['students'])
                    # check if teacher has to retire
                    if course['teacher'].info['age'] >= course['teacher'].age_high * self.model.steps_per_year:
                        course['teacher'].evolve(Pensioner)
                        missing_teachers_keys.add(c_id)

            # define set of teachers that are left with an empty class
            jobless_teachers_keys = {key for key in self.grouped_studs if key not in updated_groups}
            # define reallocating pairs dict {jobless_key: missing_key}
            # to reassign school jobless teachers to courses with students but without teacher
            pairs = {}
            for jlt_key in jobless_teachers_keys:
                for mt_key in missing_teachers_keys:
                    if abs(jlt_key - mt_key) < max_course_dist and mt_key not in pairs.values():
                        pairs.update({jlt_key: mt_key})
            for (jlt_key, mt_key) in pairs.items():
                updated_groups[mt_key]['teacher'] = self.grouped_studs[jlt_key]['teacher']
                # update sets after reallocation
                jobless_teachers_keys.remove(jlt_key)
                missing_teachers_keys.remove(mt_key)

            # check if there are still jobless teachers left in school
            if jobless_teachers_keys:
                # TODO : for jobless teachers, find another occupation. Use them when needed
                for jlt_key in jobless_teachers_keys:
                    self.grouped_studs[jlt_key]['teacher'].loc_info['course_key'] = None

            self.grouped_studs = updated_groups

            # check if there are still missing teachers for some courses and hire them
            if missing_teachers_keys:
                self.hire_teachers(missing_teachers_keys)

            # assign course id to each student
            for (c_id, course) in self.grouped_studs.items():
                for st in course['students']:
                    st.loc_info['course_key'] = c_id


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

    def hire_teachers(self, courses_keys):
        """ Hire teachers for the specified courses
            Args:
                * courses_keys: list of integer(s). Identifies courses for which teachers are missing
                    through years of age of its students
        """
        hired_teachers = self.find_teachers(courses_keys, ret_output=True)
        # assign class key to teachers and add teachers to grouped studs
        # TODO : sort employees by lang competence from lowest to highest
        # TODO : set conditions for hiring according to students age. Higher age, higher requirements
        for (k, t) in zip(courses_keys, hired_teachers):
            if isinstance(t, Teacher):
                self.grouped_studs[k]['teacher'] = t
                t.loc_info['job'] = self
                t.loc_info['course_key'] = k
                self.info['employees'].add(t)
            else:
                # turn hired agent into Teacher
                new_t = t.evolve(Teacher, ret_output=True)
                self.grouped_studs[k]['teacher'] = new_t
                new_t.loc_info['job'] = self
                new_t.loc_info['course_key'] = k
                self.info['employees'].add(new_t)

    def assign_stud(self, student):
        super().assign_stud(student)
        student.loc_info['school'] = self

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
            st.evolve(YoungUniv, university=univ)
        # rest of last year students to job market
        job_stds = set(studs).difference(set(univ_stds))
        for st in job_stds:
            st.evolve(Young)

    def swap_teachers_courses(self):
        """ Every n years on average teachers are swapped between classes """
        # sort dict keys before iteration
        sorted_keys = sorted(list(self.grouped_studs.keys()))
        # swap teacher keys, and teachers for each pair of classes
        for (k1, k2) in zip(*[iter(sorted_keys)] * 2):
            self.grouped_studs[k1]['teacher'].loc_info['course_key'] = k2
            self.grouped_studs[k2]['teacher'].loc_info['course_key'] = k1
            (self.grouped_studs[k1]['teacher'],
             self.grouped_studs[k2]['teacher']) = (self.grouped_studs[k2]['teacher'],
                                                   self.grouped_studs[k1]['teacher'])

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

    def hire_teachers(self, courses_keys):
        """ Hire teachers for the specified courses
            Args:
                * courses_keys: list of integer(s). Identifies courses for which teachers are missing
                    through years of age
        """
        hired_teachers = self.find_teachers(courses_keys, ret_output=True)
        # assign class key to teachers and add teachers to grouped studs
        # TODO : sort employees by lang competence from lowest to highest
        for (k, t) in zip(courses_keys, hired_teachers):
            # turn hired agent into Teacher
            new_t = t.evolve(TeacherUniv, ret_output=True)
            self.grouped_studs[k]['teacher'] = new_t
            new_t.loc_info['job'] = [self.univ, self.info['type']]
            new_t.loc_info['course_key'] = k
            self.info['employees'].add(new_t)
            self.univ.info['employees'].add(new_t)

    def exit_studs(self, studs):
        for st in list(studs):
            st.evolve(Young)

    def assign_stud(self, student):
        super().assign_stud(student)
        self.univ.info['students'].add(student)
        student.loc_info['university'] = [self.univ, self.info['type']]
        student.loc_info['course_key'] = int(student.info['age'] / self.model.steps_per_year)

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
    def __init__(self, clust, pos, num_places, skill_level=0, lang_policy=None):
        self.clust = clust
        self.pos = pos
        self.num_places=num_places
        self.info = {'employees': set(), 'lang_policy': lang_policy,
                     'skill_level': skill_level }
        self.agents_in = set()

    def look_for_employees(self):
        pass

    def hire_employee(self, agent):
        if agent.info['language'] in self.info['lang_policy']:
            self.num_places -= 1
            agent.loc_info['job'] = self
            self.info['employees'].add(agent)
            # TODO : check if home update needed for hired employee and their family ( school, consort job)

    # TODO : update workers by department and send them to retirement when age reached

    def __repr__(self):
        return 'Job_{0.clust!r}_{0.pos!r}'.format(self)


class Store:
    pass