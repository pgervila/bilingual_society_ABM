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
                * grown_agent: class instance. Agent that will replace fromer agent in home occupants
        """

        self.info['occupants'].remove(agent)
        if replace:
            self.info['occupants'].add(grown_agent)
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

    def find_teachers(self, courses_keys):
        """
        Hire teachers for the specified courses
            Args:
                * courses_keys: list of integer(s). Identifies courses for which teachers are missing
                    through years of age
                * ret_output: boolean. True when output needs to be returned
        """
        # TODO : when a teacher dies or retires =>
        # TODO a signal needs to be sent to school so that it can update => Call 'look_for_teachers'
        num_needed_teachers = len(courses_keys)
        school_clust_ix = self.info['clust']
        hired_teachers = []

        # first check among teachers employed at school but without course_key assigned
        # TODO : add more restrictive conditions for higher courses
        free_staff_from_school = [ag for ag in self.info['employees'] if not ag.loc_info['job'][1]]
        hired_teachers.extend(free_staff_from_school)
        if len(hired_teachers) >= num_needed_teachers:
            hired_teachers = set(hired_teachers[:num_needed_teachers])
            return hired_teachers

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
                        # keep teacher if he/she does not have to retire
                        if self.grouped_studs[c_id + 1]['teacher']:
                            if self.grouped_studs[c_id + 1]['teacher'].info['age'] < retirement_age:
                                updated_groups[c_id + 1]['teacher'] = self.grouped_studs[c_id + 1]['teacher']
                            else:
                                updated_groups[c_id + 1]['teacher'] = None
                        else:
                            updated_groups[c_id + 1]['teacher'] = None
                else:
                    self.exit_studs(course['students'])

                # check if teacher has to retire
                if course['teacher'].info['age'] >= course['teacher'].age_high * self.model.steps_per_year:
                    course['teacher'].evolve(Pensioner)

            # define set of teachers that are left with an empty class
            jobless_teachers_keys = {key for key in self.grouped_studs
                                     if key not in updated_groups and self.grouped_studs[key]['teacher']}

            # define set of missing teachers in updated_groups
            missing_teachers_keys = {key for key in updated_groups
                                     if not updated_groups[key]['teacher']}

            # define reallocating pairs dict {jobless_key: missing_key}
            # to reassign school jobless teachers to courses with students but without teacher
            pairs = {}
            for jlt_key in list(jobless_teachers_keys):
                for mt_key in list(missing_teachers_keys):
                    if abs(jlt_key - mt_key) < max_course_dist and mt_key not in pairs.values():
                        pairs.update({jlt_key: mt_key})
            for (jlt_key, mt_key) in pairs.items():
                updated_groups[mt_key]['teacher'] = self.grouped_studs[jlt_key]['teacher']
                updated_groups[mt_key]['teacher'].loc_info['job'][1] = mt_key
                # update sets after reallocation
                jobless_teachers_keys.remove(jlt_key)
                missing_teachers_keys.remove(mt_key)

            # check if there are still jobless teachers left in school
            if jobless_teachers_keys:
                # TODO : for jobless teachers, find another occupation. Use them when needed
                for jlt_key in jobless_teachers_keys:
                    self.grouped_studs[jlt_key]['teacher'].loc_info['job'][1] = None

            # assign update groups to grouped_studs class attribute
            self.grouped_studs = updated_groups

            # check if there are still missing teachers for some courses and hire them
            if missing_teachers_keys:
                self.hire_teachers(missing_teachers_keys)

    def remove_student_from_course(self, student, educ_center, replace=None, grown_agent=None):
        """ remove students from their course, and optionally replace them with grown_agents
            Args:
                * student: agent instance
                * educ_center: string. It is either 'school' or 'university'
                * replace:boolean. If True, grown_agent msut be specified
                * grown_agent: agent instance
        """
        course_key = student.loc_info[educ_center][1]
        self.grouped_studs[course_key]['students'].remove(student)
        if replace and educ_center == 'school' and not isinstance(student, Adolescent):
            self.grouped_studs[course_key]['students'].add(grown_agent)

    def remove_teacher_from_course(self, teacher, replace=None, new_teacher=None):
        if teacher.loc_info['job'][1]:
            course_key = teacher.loc_info['job'][1]
            self.grouped_studs[course_key]['teacher'] = None
            if replace:
                self.grouped_studs[course_key]['teacher'] = new_teacher

    def remove_agent_in(self, agent):
        if agent in self.agents_in:
            self.agents_in.remove(agent)

    def remove_course(self, course_key):
        """ Remove school course if no students are left in it. Relocate teacher"""
        if not self[course_key]['students']:
            courseless_teacher = self[course_key]['teacher']
            courseless_teacher.loc_info['job'][1] = None
            del self[course_key]

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

    def assign_teacher(self, teacher, course_key):
        self.grouped_studs[course_key]['teacher'] = teacher
        teacher.loc_info['job'][1] = course_key
        if teacher not in self.info['employees']:
            teacher.loc_info['job'][0] = self
            self.info['employees'].add(teacher)

    def hire_teachers(self, courses_keys):
        """ Hire teachers for the specified courses
            Args:
                * courses_keys: list of integer(s). Identifies courses for which teachers are missing
                    through years of age of its students
        """
        hired_teachers = self.find_teachers(courses_keys)
        # assign class key to teachers and add teachers to grouped studs
        # TODO : sort employees by lang competence from lowest to highest
        # TODO : set conditions for hiring according to students age. Higher age, higher requirements
        for (k, t) in zip(courses_keys, hired_teachers):
            if isinstance(t, Teacher):
                self.grouped_studs[k]['teacher'] = t
                t.loc_info['job'] = [self, k]
                self.info['employees'].add(t)
            else:
                # turn hired agent into Teacher
                new_t = t.evolve(Teacher, ret_output=True)
                self.grouped_studs[k]['teacher'] = new_t
                new_t.loc_info['job'] = [self, k]
                self.info['employees'].add(new_t)

    def assign_stud(self, student):
        super().assign_stud(student)
        course_key = int(student.info['age'] / self.model.steps_per_year)
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
            self.grouped_studs[k1]['teacher'].loc_info['job'][1] = k2
            self.grouped_studs[k2]['teacher'].loc_info['job'][1] = k1
            (self.grouped_studs[k1]['teacher'],
             self.grouped_studs[k2]['teacher']) = (self.grouped_studs[k2]['teacher'],
                                                   self.grouped_studs[k1]['teacher'])

    def remove_student(self, student, replace=False, grown_agent=None):
        self.info['students'].remove(student)
        # replace agent only if it is not an adolescent
        if replace and not isinstance(student, Adolescent):
            self.info['students'].add(grown_agent)
        # course_key
        self.remove_student_from_course(student, 'school', replace=replace, grown_agent=grown_agent)
        self.remove_agent_in(student)

    def remove_teacher(self, teacher, replace=False, new_teacher=None):
        self.info['employees'].remove(teacher)
        if replace:
            self.info['employees'].add(new_teacher)
        # course_key
        self.remove_teacher_from_course(teacher, replace=replace, new_teacher=new_teacher)
        self.remove_agent_in(teacher)

    def update_courses(self, max_course_dist=3):
        super().update_courses()
        # assign course id to each student
        for (c_id, course) in self.grouped_studs.items():
            for st in course['students']:
                st.loc_info['school'][1] = c_id




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
            new_t = t.evolve(TeacherUniv, ret_output=True)
            self.grouped_studs[k]['teacher'] = new_t
            new_t.loc_info['job'] = [self.univ, k, self.info['type']]
            self.info['employees'].add(new_t)
            self.univ.info['employees'].add(new_t)

    def exit_studs(self, studs):
        for st in list(studs):
            st.evolve(Young)

    def assign_stud(self, student):
        super().assign_stud(student)

        self.univ.info['students'].add(student)
        course_key = int(student.info['age'] / self.model.steps_per_year)
        student.loc_info['university'] = [self.univ, course_key, self.info['type']]

    def remove_student(self, student):
        # remove from uni and fac
        self.univ.info['students'].remove(student)
        self.info['students'].remove(student)
        # course_key
        self.remove_student_from_course(student, 'university')
        self.remove_agent_in(student)

    def remove_teacher(self, teacher, replace=False, new_teacher=None):
        self.univ.info['employees'].remove(teacher)
        self.info['employees'].remove(teacher)
        # course_key
        self.remove_teacher_from_course(teacher, replace=replace, new_teacher=new_teacher)
        self.remove_agent_in(teacher)

    def update_courses(self, max_course_dist=3):
        super().update_courses()
        # assign course id to each student
        for (c_id, course) in self.grouped_studs.items():
            for st in course['students']:
                st.loc_info['university'][1] = c_id

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

    def look_for_employee(self):
        """ Look for employee that meets requirements """

        # look for suitable agents in any cluster
        for ag in self.model.schedule.agents:
            if ag.info['language'] in self.info['lang_policy'] and isinstance(ag, Young) and not isinstance(ag, Teacher):
                self.hire_employee(ag)
                break

    def hire_employee(self, agent, move_home=False):
        if agent.info['language'] in self.info['lang_policy']:
            self.num_places -= 1
            agent.loc_info['job'] = self
            self.info['employees'].add(agent)
            if move_home:
                agent.move_to_new_home(marriage=False)

    # TODO : update workers by department and send them to retirement when age reached

    def remove_employee(self, agent, replace=None, new_agent=None):
        self.info['employees'].remove(agent)
        if replace:
            self.info['employees'].add(new_agent)
        if agent in self.agents_in:
            self.agents_in.remove(agent)

    def __repr__(self):
        return 'Job_{0.clust!r}_{0.pos!r}'.format(self)


class Store:
    pass

class BookStore(Store):
    pass