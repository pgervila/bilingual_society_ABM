import networkx as nx

class Networks:
    """ Class to deal with evrything that has to do with networks in model"""

    def __init__(self, model):
        self.model = model

    def define_lang_interaction(self, ag1, ag2, ret_pcts=False):
        # find out lang of interaction btw family members
        # consorts
        pct11 = ag1.lang_stats['L1']['pct'][ag1.age]
        pct12 = ag1.lang_stats['L2']['pct'][ag1.age]
        pct21 = ag2.lang_stats['L1']['pct'][ag2.age]
        pct22 = ag2.lang_stats['L2']['pct'][ag2.age]
        if (ag1.language, ag2.language) in [(0, 0), (0, 1), (1, 0)]:
            lang = 0
        elif (ag1.language, ag2.language) in [(2, 1), (1, 2), (2, 2)]:
            lang = 1
        elif (ag1.language, ag2.language) == (1, 1):
            # Find weakest combination lang-agent, pick other language as common one
            idx_weakest = np.argmin([pct11, pct12, pct21, pct22])
            if idx_weakest in [0, 2]:
                lang = 1
            else:
                lang = 0
        if ret_pcts:
            return lang, [pct11, pct12, pct21, pct22]
        else:
            return lang

    def define_family_networks(self):
        # Method to define families and also adds relatives to known_people_network
        # marriage, to make things simple, only allowed for combinations  0-1, 1-1, 1-2
        for clust_idx, clust_info in self.model.clusters_info.items():
            for idx, family in enumerate(zip(*[iter(clust_info['agents'])] * self.model.family_size)):
                # set ages of family members
                steps_per_year = 36
                min_age, max_age = 40 * steps_per_year, 50 * steps_per_year
                family[0].age, family[1].age = np.random.randint(min_age, max_age, size=2)
                min_age, max_age = 10 * steps_per_year, 20 * steps_per_year
                family[2].age, family[3].age = np.random.randint(min_age, max_age, size=2)

                # assign same home to all family members
                home = clust_info['homes'][idx]
                # import ICs and assign home
                # apply correlation between parents' and children's lang knowledge if parents bilinguals
                if 1 in [family[0].language, family[1].language]:
                    key_parents = [] # define list to store parents' percentage knowledge
                    for ix, member in enumerate(family):
                        if ix <2 and member.language == 1:
                            key = np.random.choice(self.model.ic_pct_keys)
                            key_parents.append(key)
                            member.set_lang_ics(biling_key=key)
                        elif ix < 2:
                            lang_mono = member.language
                            member.set_lang_ics()
                        elif ix >= 2:
                            if len(key_parents) == 1:
                                if not lang_mono: # mono in lang 0
                                    key = (key_parents[0] + 100) / 2
                                else: # mono in lang 1
                                    key = key_parents[0] / 2
                            else:
                                key = sum(key_parents) / len(key_parents)
                            key = self.model.ic_pct_keys[
                                bisect.bisect_left(self.model.ic_pct_keys, key, hi=len(self.model.ic_pct_keys)-1)
                            ]
                            member.set_lang_ics(biling_key=key)
                        self.model._assign_home_to_agent(member, home)
                else: # monolingual parents
                    # check if children are bilingual
                    if 1 in [family[2].language, family[3].language]:
                        for ix, member in enumerate(family):
                            if ix < 2:
                                member.set_lang_ics()
                            else:
                                if member.language == 1:
                                    # logical that child has much better knowledge of parents lang
                                    member.set_lang_ics(biling_key=90)
                                else:
                                    member.set_lang_ics()
                            self.model._assign_home_to_agent(member, home)
                    else:
                        for member in family:
                            member.set_lang_ics()
                            self.model._assign_home_to_agent(member, home)

                # assign job to parents
                for parent in family[:2]:
                    while True:
                        job = np.random.choice(clust_info['jobs'])
                        if job.num_places:
                            job.num_places -= 1
                            parent.loc_info['job'] = job
                            job.employees.add(parent)
                            break
                # assign school to children
                # find closest school
                idx_school = np.argmin([pdist([home.pos, school.pos])
                                        for school in clust_info['schools']])
                school = clust_info['schools'][idx_school]
                for child in family[2:]:
                    child.loc_info['school'] = school
                    school.students.add(child)
                # find out lang of interaction btw family members
                # consorts
                lang_consorts, pcts = self.model.define_lang_interaction(family[0], family[1], ret_pcts=True)
                # language of children with father, mother
                lang_with_father = np.argmax(pcts[:2])
                lang_with_mother = np.argmax(pcts[2:])
                # siblings
                avg_lang = (lang_with_father + lang_with_mother) / 2
                if avg_lang == 0:
                    lang_siblings = 0
                elif avg_lang == 1:
                    lang_siblings = 1
                else:
                    lang_siblings = self.model.define_lang_interaction(family[2], family[3])

                    # find weakest, pick opposite PREVIOUS implem
                    # pct11 = family[2].lang_stats['L1']['pct'][family[2].age]
                    # pct12 = family[2].lang_stats['L2']['pct'][family[2].age]
                    # pct21 = family[3].lang_stats['L1']['pct'][family[3].age]
                    # pct22 = family[3].lang_stats['L2']['pct'][family[3].age]
                    # idx_weakest = np.argmin([pct11, pct12, pct21, pct22])
                    # if idx_weakest in [0, 2]:
                    #     lang_siblings = 1
                    # else:
                    #     lang_siblings = 0

                # add family edges in family and known_people networks ( both are DIRECTED networks ! )
                for (i, j) in [(0, 1), (1, 0)]:
                    self.model.family_network.add_edge(family[i], family[j], fam_link='consort', lang=lang_consorts)
                    self.model.known_people_network.add_edge(family[i], family[j], family=True, lang=lang_consorts)
                for (i, j, link) in [(0, 2, 'child'), (2, 0, 'father'), (0, 3, 'child'), (3, 0, 'father')]:
                    self.model.family_network.add_edge(family[i], family[j], fam_link=link, lang=lang_with_father)
                    self.model.known_people_network.add_edge(family[i], family[j], family=True, lang=lang_with_father)
                for (i, j, link) in [(1, 2, 'child'), (2, 1, 'mother'), (1,3, 'child'), (3,1,'mother')]:
                    self.model.family_network.add_edge(family[i], family[j], fam_link=link, lang=lang_with_mother)
                    self.model.known_people_network.add_edge(family[i], family[j], family=True, lang=lang_with_mother)
                for (i, j) in [(2, 3), (3, 2)]:
                    self.model.family_network.add_edge(family[i], family[j], fam_link='sibling', lang=lang_siblings)
                    self.model.known_people_network.add_edge(family[i], family[j], family=True, lang=lang_siblings)

            # set up agents left out of family partition of cluster
            len_clust = len(clust_info['agents'])
            num_left_agents = len_clust % self.model.family_size
            if num_left_agents:
                for ag in clust_info['agents'][-num_left_agents:]:
                    min_age, max_age = 40 * steps_per_year, 60 * steps_per_year
                    ag.age = np.random.randint(min_age, max_age)
                    ag.set_lang_ics()
                    home = clust_info['homes'][idx + 1]
                    ag.loc_info['home'] = home
                    home.agents_in.add(ag)
                    while True:
                        job = np.random.choice(clust_info['jobs'])
                        if job.num_places:
                            job.num_places -= 1
                            ag.loc_info['job'] = job
                            break

    def define_friendship_networks(self):
        # TODO :
        # Apply small world graph to relevant nodes using networkx
        friends_per_agent = np.random.randint(1, 5, size=self.model.num_people)
        for ag, num_friends in zip(self.model.schedule.agents, friends_per_agent):
            if ag.loc_info['job'] and len(self.model.friendship_network[ag]) < num_friends:
                ag_occupation = ag.loc_info['job']
                colleagues = 'employees'
            elif ag.loc_info['school'] and len(self.model.friendship_network[ag]) < num_friends:
                ag_occupation = ag.loc_info['school']
                colleagues = 'students'
            else:
                continue
            for coll in getattr(ag_occupation, colleagues).difference({ag}):
                #check colleague lang distance and all frienship conditions
                if (abs(coll.language - ag.language) <= 1 and
                len(self.model.friendship_network[coll]) < friends_per_agent[coll.unique_id] and
                coll not in self.model.friendship_network[ag] and
                coll not in self.model.family_network[ag]):
                    lang = self.model.define_lang_interaction(ag, coll)
                    self.model.friendship_network.add_edge(ag, coll, lang=lang)
                    # kpnetwork is directed graph !
                    self.model.known_people_network.add_edge(ag, coll, friends=True, lang=lang)
                    self.model.known_people_network.add_edge(coll, ag, friends=True, lang=lang)
                if len(self.model.friendship_network[ag]) > num_friends - 1:
                    break

