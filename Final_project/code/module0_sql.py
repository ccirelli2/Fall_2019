# DOCUMENTATION ----------------------------------------------
'''
1.) Note:       What if the strokes are not of the same length?

'''




# FUNCTIONS --------------------------------------------------

def sql_insert_stroke_info(mydb, mycursor, df):
    '''
    Input:      We're going to receive a dataframe for each stroke file
                Then we will iterate the df as a tuple and insert
                them one by one into the db
    '''
    # Logging
    print('Starting Insertion Process')
    
    # SQL Insert Statement
    sql = '''INSERT IGNORE INTO stroke_info (form_id, writer_id, stroke, y, x, t)
             VALUES (%s, %s, %s, %s, %s, %s)'''
   

    # Iterate DataFrame
    for row in df.itertuples():
        val = (str(row[1]), str(row[2]), str(row[3]), str(row[4]), str(row[5]), str(row[6]))
        mycursor.execute(sql, val)
        mydb.commit()

    # Logging
    print('Finished Insertion \n')

    # Return None
    return None



def sql_insert_author(mydb, mycursor, val):
    '''
    Input:  val is the list of values to be inserted
    '''

    sql = '''INSERT IGNORE INTO author_info ( name, dob, education, gender, country_origin, 
                                       native_language, writing_hand)
                              VALUES (%s, %s, %s, %s, %s, %s, %s)'''

    mycursor.execute(sql, val)
    mydb.commit()

    '''
    Input:  Entire path to xml file
    ET      use to parse xml file into tree structure
    tag     will give you all the tags in the file
    attrib  attributes
    Example:
    {'name': '10208', 'DayOfBirth': '1971-06-03', 'EducationalDegree': 'Dipl. ing.',
    'Gender': 'Male', 'NativeCountry': 'Switzerland', 'NativeLanguage': 'Swiss German',
    'Profession': 'Student', 'Science': 'Computer Science', 'WritingType': 'Right-handed',
    'WrittenLanguage': 'German'}
    '''
    # Parse XML File
    tree    = ET.parse(xml_file)
    root    = tree.getroot()


    # Iterate Tree:
    for child in root:

        # Add Try Statement as there may be errors
        try:
            attrib          = child.attrib
            name            = attrib['name']
            dob             = attrib['DayOfBirth']
            education       = attrib['EducationalDegree']
            gender          = attrib['Gender']
            country_origin  = attrib['NativeCountry']
            native_lang     = attrib['NativeLanguage']
            writing_hand    = attrib['WritingType']

            # Create Date Object for Date of Birth
            try:
                dob = datetime.strptime(dob, '%Y-%m-%d')
            except ValueError as err:
                print('Error => {}'.format(err))
                dob = datetime.strptime('1900-01-01', '%Y-%m-%d')

            # Define Val Object
            val = (name, dob, education, gender, country_origin, native_lang, writing_hand)

            # Insert Values Into Table
            sql_insert_author(mydb, mycursor, val)

        # Except Error & Return to Stdout
        except KeyError as err:
            print('Key Error  => {}'.format(err))

    # Logging
    print('Insertion Complete')

    # Return None
    return None


# BUILD TABLE - STROKE FILES


def build_database_online_files_sql(root_dir_data):
    ''' 
    Description:    Iterate entire root directory where the datafiles are located
                    parse the xml files & insert them into a mysql table. 
    '''


    # Get List of All XML Files In Dir
    path2files = []

    # Iterate Recursively Entire Directory
    for root, dirs, files in os.walk(root_dir_data):
        # Iterate List of Directories
        for dir_ in dirs:
            # Get Path 2 File
            path2file = root + '/' + dir_
            afile = os.listdir(path2file)[0]
            # If the file listed here is a strokes file
            if afile == 'strokesz.xml':
                # Create path2file and append to list
                path2file = path2file + '/' + afile
                path2files.append(path2file)
            # IF not, then there is another sudirectory that we need to take into consideration
            else:
                subdir = afile
                afile  = os.listdir(path2file + '/' + subdir)[0]
                path2file = path2file + '/' + subdir + '/' + afile
                path2files.append(path2file)

    # Iterate List Path2files
    for path in path2files:
        # Start Timer
        start = datetime.now()
        # Parse XML File
        df = m1.parse_stroke_file(path)
        # Insert XML File Into Database
        m0.sql_insert_stroke_info(mydb, mycursor, df)
        end = datetime.now()
        # Logging time
        print('Process time => {}\n'.format(end-start))

    # Return None
    return None