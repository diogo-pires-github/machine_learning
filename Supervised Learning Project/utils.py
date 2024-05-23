import pandas as pd
import numpy as np
import glob
import re
import os

from sklearn.base import BaseEstimator, TransformerMixin


def get_random_base(y):
    '''
    Get weighted random numbers, based on distribution of classes of given y.
    '''
    s = pd.Series(y).value_counts(normalize=True)
    d = {}
    for i in range(s.shape[0]):
        d[s.index[i]] = s.values[i]

    y_base = np.array([np.random.choice(list(d.keys()), p=list(d.values())) for _ in range(y.shape[0])])
    return y_base


def get_submission_data(trained_model, preprocessed_test_df, test_df_encounter_id, save_path='.'):

    '''
    Gets preprocessed test data, predicts with the given model, parses the prediction and saves to givel save_path.
    The output csv will be named 'submission_v<x>.csv', where <x> will be the last submission number +1.
    '''
    if isinstance(preprocessed_test_df, pd.DataFrame):
        if 'encounter_id' in preprocessed_test_df.columns:
            preprocessed_test_df = preprocessed_test_df.set_index('encounter_id')
            test_df_encounter_id = preprocessed_test_df.index.to_list()

    y_pred = trained_model.predict(preprocessed_test_df)
    sub_df = pd.DataFrame({'encounter_id': test_df_encounter_id, 'readmitted_binary': y_pred})
    sub_df = sub_df.set_index('encounter_id')
    if sub_df.readmitted_binary.dtype != 'O':
        sub_df = sub_df.readmitted_binary.map({0: 'No', 1: 'Yes'})

    list_of_files = glob.glob('../submissions/*.csv') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    latest_version = int(re.findall(r'\d+', latest_file)[0])
    to_submit_path = os.path.join(save_path, f'submission_v{latest_version+1}.csv')

    sub_df.to_csv(to_submit_path)

    print(f'{to_submit_path} was created! Good luck on the score!')


class CustomCleaner(BaseEstimator, TransformerMixin):
    '''
    Custom class to apply transformations to the dataset and creates new features. 
    The following transformations are performed:
        - All "?" values are mapped as np.nan
        - medications: is parsed and split into a dummy for each unique medication
        - age: is transformed from categorical to numerical, using the middle point of the category. (ie,[0-10) becomes 5)
        - gender: 'Unknown/Invalid' values are mapped as np.nan
        - discharge_disposition: reduces all the categories to the following ones:
            * Home;
            * Home with medical care;
            * Expired;
            * Healthcare facility;
            * Unknown;
            * Other
        - admission_source: reduces all the categories to the following ones:
            * Emergency room;
            * Transfered;
            * Referral;
            * Unknown admission;
            * Other
        - payer_code: assumes np.nan means "no insurance", mapping False to np.nan or True otherwise
        - admission_type: reduces all the categories to the following ones:
            * Emergency;
            * Elective;
            * Urgent;
            * Other
        - medical_specialty: reduces all the categories to the following ones:
            * Allergy and immunology;
            * Anesthesiology;
            * Diagnostic radiology;
            * Emergency medicine;
            * Internal medicine;
            * Neurology;
            * Obstetrics and gynecology;
            * Ophthalmology;
            * Pathology;
            * Pediatrics;
            * Physical medicine and rehabilitation;
            * Psychiatry;
            * Radiation oncology;
            * Surgery;
            * Oncology;
            * Other
        - primary_diagnosis, secondary_diagnosis, additional_diagnosis: uses the given ICD-9 codes to map into the following categories:
            * Circulatory;
            * Respiratory;
            * Gastrointestinal;
            * Diabetes;
            * Injury;
            * Musculoskeletical;
            * Genitourinary;
            * Neoplasms;
            * Other
    The following features are created:
        - several medications: for each unique medication in medications column, a dummy feature is created
        - number_diab_meds: count of the dummy medications columns along same observation
        - number_visits_per_patient: count of total times the same patient appeared in the dataset
        - n_medication_per_day: average of number of medications the patient took per day in hospital
        - n_lab_tests_per_day: average of number of lab tests the patient had per day in hospital
        - prop_inpatient, prop_outpatient, prop_emergency: proportions of each type of visit to the total number of visits 
    '''

    def __init__(self):
        super().__init__()
        
        self.unique_meds = []
        self.age_mapping = {
                            '[0-10)': 5,
                            '[10-20)': 15,
                            '[20-30)': 25,
                            '[30-40)': 35,
                            '[40-50)': 45,
                            '[50-60)': 55,
                            '[60-70)': 65,
                            '[70-80)': 75,
                            '[80-90)': 85,
                            '[90-100)': 95
                            }
        self.dx_cols = ['primary_diagnosis', 'secondary_diagnosis', 'additional_diagnosis']
        self.insurances_mapping = {'MC':True,'HM':True,'UN':True,'SP':True,'SI':True,'CM':True,
                           'DM':True,'CP':True,'MD':True,'OG':True,'BC':True,'PO':True,
                           'WC':True,'OT':True,'MP':True,'CH':True,'FR':True,'?':False, np.nan:False}
        
    
    def fit(self, X, y=None):
        # Getting the unique meds in the dataset
        X.medication = X.medication.str.replace("'", '').str.replace('[', '').str.replace(']', '').str.replace(' ', '')

        for meds in X.medication.unique().tolist():
            self.unique_meds.extend(meds.split(','))

        self.unique_meds = list(set(self.unique_meds))
        if '' in self.unique_meds:
            self.unique_meds.remove('')

        self.patient_ids = {}

        return self

    
    def transform(self, X, y=None):
        X_2 = X.copy()

        # Getting all the medications from medication
        X_2 = self._create_meds(X_2)

        # Age to numeric
        X_2.age.replace(self.age_mapping, inplace=True)

        # Replacing ? by nan
        X_2.replace('?', np.nan, inplace=True)

        # Replacing Unknow gender by nan
        X_2.loc[~(X.gender == 'Male') & ~(X.gender == 'Female'), 'gender'] = np.nan

        # Cleaning discharge_disposition
        X_2['discharge_disposition'] = X_2['discharge_disposition'].apply(self._categorize_discharge)

        # Cleaning admission_source
        X_2['admission_source'] = X_2['admission_source'].apply(self._categorize_admission)

        # Mapping payer_code
        X_2['payer_code'] = X_2.payer_code.replace(self.insurances_mapping)	

        # Cleaning admission_type
        X_2.loc[~(X_2.admission_type == 'Emergency') & ~(X_2.admission_type == 'Elective') & ~(X_2.admission_type == 'Urgent'), 'admission_type'] = 'Other'

        # Getting number of diab medications
        X_2['number_diab_meds'] = X_2.loc[:, ['metformin-rosiglitazone', 'troglitazone', 'tolazamide',
       'metformin-pioglitazone', 'tolbutamide', 'pioglitazone',
       'glimepiride-pioglitazone', 'nateglinide', 'glyburide-metformin',
       'insulin', 'glimepiride', 'glyburide', 'acarbose', 'metformin',
       'glipizide', 'chlorpropamide', 'glipizide-metformin', 'miglitol',
       'repaglinide', 'rosiglitazone']].sum(axis=1)

        # Cleaning medical specialty
        X_2['medical_specialty'] = X_2['medical_specialty'].apply(self._categorize_specialty)

        # Transforming _diagnosis columns
        X_2 = self._transform_diagnosis(X_2)

        # Getting visits per patient
        X_2 = self._get_count_pt_visit(X_2)
        
        # Getting avg of medication taken per day in hospital
        X_2['n_medication_per_day'] = np.log(X_2['number_of_medications'] / X_2['length_of_stay_in_hospital'])

        # Getting avg of lab tests made per day in hospital
        X_2['n_lab_tests_per_day'] = np.log(X_2['number_lab_tests'] / X_2['length_of_stay_in_hospital'])

        # Getting proportion of type of visit 
        X_2['prop_inpatient'] = X_2['inpatient_visits_in_previous_year'] / X_2['number_visits_per_patient']
        X_2['prop_outpatient'] = X_2['outpatient_visits_in_previous_year'] / X_2['number_visits_per_patient']
        X_2['prop_emergency'] = X_2['emergency_visits_in_previous_year'] / X_2['number_visits_per_patient']

        return X_2


    def _categorize_specialty(self, specialty):
        if specialty in ['AllergyandImmunology', 'Pediatrics-AllergyandImmunology']:
            return 'Allergy and immunology'
        elif specialty in ['Anesthesiology-Pediatric', 'Anesthesiology']:
            return 'Anesthesiology'
        elif specialty in ['Radiologist', 'Radiology']:
            return 'Diagnostic radiology'
        elif specialty in ['Pediatrics-EmergencyMedicine', 'Family/GeneralPractice', 'Emergency/Trauma']:
            return 'Emergency medicine'
        elif specialty in ['InternalMedicine', 'SportsMedicine', 'Hematology', 'Nephrology', 'Cardiology', 'Gastroenterology', 'Pulmonology', 'Hematology/Oncology', 'Endocrinology', 'Endocrinology-Metabolism', 'Otolaryngology', 'InfectiousDiseases', 'Rheumatology', 'Proctology']:
            return 'Internal medicine'
        elif specialty in ['Neurology', 'Neurophysiology']:
            return 'Neurology'
        elif specialty in ['Gynecology', 'ObstetricsandGynecology', 'Obsterics&Gynecology-GynecologicOnco', 'Obstetrics']:
            return 'Obstetrics and gynecology'
        elif specialty in ['Ophthalmology']:
            return 'Ophthalmology'
        elif specialty in ['Pathology']:
            return 'Pathology'
        elif specialty in ['Pediatrics', 'Pediatrics-Endocrinology', 'Pediatrics-CriticalCare', 'Pediatrics-InfectiousDiseases', 'Pediatrics-Hematology-Oncology', 'Pediatrics-Neurology', 'Pediatrics-Pulmonology', 'Cardiology-Pediatric']:
            return 'Pediatrics'
        elif specialty in ['PhysicalMedicineandRehabilitation', 'DCPTEAM', 'Osteopath']:
            return 'Physical medicine and rehabilitation'
        elif specialty in ['Psychiatry-Child/Adolescent', 'Psychiatry', 'Psychology']:
            return 'Psychiatry'
        elif specialty in ['Oncology']:
            return 'Radiation oncology'
        elif specialty in ['Surgery-General', 'Surgery-Neuro', 'Surgery-Vascular', 'Surgery-Thoracic', 'Surgery-Cardiovascular', 'SurgicalSpecialty', 'Surgery-Plastic', 'Surgery-Maxillofacial', 'Surgery-Cardiovascular/Thoracic', 'Surgeon', 'Surgery-Colon&Rectal', 'Surgery-Pediatric', 'Orthopedics', 'Orthopedics-Reconstructive']:
            return 'Surgery'
        elif specialty in ['Urology']:
            return 'Urology'
        elif specialty in ['PhysicianNotFound','Podiatry','Hospitalist','OutreachServices','Dentistry','Speech','Resident']:
            return 'Other'

    def _create_meds(self, X):
        X_2 = X.copy()

        X_2.medication = X_2.medication.str.replace("'", '').str.replace('[', '').str.replace(']', '').str.replace(' ', '')
        for med in self.unique_meds:
            X_2[med] = False

        for i, meds in X_2[['medication']].iterrows():
            list_of_meds = meds.iloc[0].split(',')
            for med in list_of_meds:
                if med != '':
                    X_2.loc[i, med] = True
                else:
                    continue

        return X_2

    def _transform_diagnosis(self, X):
        X_2 = X.copy()

        for col in self.dx_cols:
            X_2.loc[X_2[col] == '?', col] = -99
            X_2.loc[(X_2[col].str.contains('V')) | (X_2[col].str.contains('E')), col] = -99
            X_2[col] = X_2[col].astype(np.float32)

            X_2['temporary'] = np.nan

            X_2.loc[(X_2[col] >= 390) & (X_2[col] <= 459) | (X_2[col] == 785), 'temporary'] = 'Circulatory'
            X_2.loc[(X_2[col] >= 460) & (X_2[col] <= 519) | (X_2[col] == 786), 'temporary'] = 'Respiratory'
            X_2.loc[(X_2[col] >= 520) & (X_2[col] <= 579) | (X_2[col] == 787), 'temporary'] = 'Gastrointestinal'
            X_2.loc[(X_2[col] >= 250) & (X_2[col] <= 251), 'temporary'] = 'Diabetes'
            X_2.loc[(X_2[col] >= 800) & (X_2[col] <= 999), 'temporary'] = 'Injury'
            X_2.loc[(X_2[col] >= 710) & (X_2[col] <= 739), 'temporary'] = 'Musculoskeletical'
            X_2.loc[(X_2[col] >= 580) & (X_2[col] <= 629) | (X_2[col] == 788), 'temporary'] = 'Genitourinary'
            X_2.loc[(X_2[col] >= 140) & (X_2[col] <= 239) | (X_2[col] == 788), 'temporary'] = 'Neoplasms'

            X_2['temporary'] = X_2.temporary.fillna('Other')
            X_2[col] = X_2['temporary']
            X_2.drop(columns='temporary', inplace=True)

        return X_2

    def _get_count_pt_visit(self, X):
        X_2 = X.copy()

        pt_counts = X_2.groupby('patient_id').count().loc[:,'encounter_id']

        X_2['number_visits_per_patient'] = 0
        for pt, count in zip(pt_counts.index, pt_counts.values):
            if pt not in self.patient_ids.keys():
                self.patient_ids[pt] = count
            else:
                self.patient_ids[pt] += count
                
            X_2.loc[X_2.patient_id == pt, 'number_visits_per_patient'] = self.patient_ids[pt]


        return X_2

    def _categorize_discharge(self, discharge_disposition):

        if discharge_disposition in ['Discharged to home']:
            return 'Home'
        if discharge_disposition in ['Discharged/transferred to home with home health service',
                                   'Discharged/transferred to home under care of Home IV provider',
                                   'Hospice / home']:
            return 'Home with medical care'
        if discharge_disposition in ['Expired',
                                   'Expired at home. Medicaid only, hospice',
                                   'Expired in a medical facility. Medicaid only, hospice']:
            return 'Expired'
        if discharge_disposition in ['Left AMA',
                                   'Admitted as an inpatient to this hospital',
                                   'Still patient or expected to return for outpatient services']:
            return 'Other'
        if discharge_disposition in ['Not Mapped', np.nan]:
            return 'Unknown'

        return 'Healthcare facility'


    def _categorize_admission(self, admission_source):
        if admission_source in [' Emergency Room']:
            return 'Emergency Room'
        if admission_source in ['Transfer from a hospital',
                                  ' Transfer from another health care facility',
                                  ' Transfer from a Skilled Nursing Facility (SNF)',
                                  ' Transfer from hospital inpt/same fac reslt in a sep claim',
                                  ' Transfer from critial access hospital',
                                  ' Transfer from Ambulatory Surgery Center']:
            return 'Transfered'
        if admission_source in [' Physician Referral',
                                  'Clinic Referral',
                                  'HMO Referral']:
            return 'Referral'
        if admission_source in [' Court/Law Enforcement',
                                  ' Extramural Birth',
                                  'Normal Delivery',
                                  ' Sick Baby']:
            return 'Other'

        return 'Unknown Admission'
