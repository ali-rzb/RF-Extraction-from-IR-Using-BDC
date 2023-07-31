__copyright__ = """

    Copyright 2022 Ali Roozbehi

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""

import copy, json, os, inspect

class data_object_class:
    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return self.__dict__ == other.__dict__
        else:
            raise TypeError('Comparing object is not of the same type.')
    def cast(instance):
        attributes = inspect.getmembers(instance, lambda a: not(inspect.isroutine(a)))
        data = list(dict([a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]).values())
        data_columns = list(dict([a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]).keys())
        
        args = inspect.getfullargspec(type(instance))
        columns = list(args)[0]
        columns.remove('self')
        order_list = []
        for c in data_columns:
            order_list.append(columns.index(c))
        data = [x for _, x in sorted(zip(order_list, data))]
        
        return type(instance)(*data)

class db_settings:
    splitter = '#$#'

class db_info:
    def __init__(self, column_names: list, column_types: dict, col_count: int, row_count: int, has_id: bool) -> None:
        self.column_names = column_names
        self.column_types = column_types
        self.col_count = col_count
        self.row_count = row_count
        self.has_id = has_id

def get_command(msg, commands_list):
    for i in range(len(commands_list)):
        commands_list[i] = str.lower(commands_list[i])
    error_msg = 'Wrong Command, please insert'
    for i, c in enumerate(commands_list):
        if i != len(commands_list):
            error_msg = error_msg + ' ' + c + ' or'
        else:
            error_msg = error_msg + ' ' + c

    command = ''
    print(msg)
    while str.lower(command) not in commands_list:
        command = input()
        if str.lower(command) in commands_list:
            return command
        else:
            print(error_msg)

def create(path, _class, clear_create=False):
    args = inspect.getfullargspec(_class)
    columns = list(args)[0]
    if 'self' in columns:
        columns.remove('self')

    # FORBIDDEN_CHARs = ['#', '$', '<', '>']
    FORBIDDEN_CHARs = list(set(list(db_settings.splitter))) + [db_settings.splitter]
    error_msg = ''
    for c in columns:
        if any(item in c for item in FORBIDDEN_CHARs):
            error_msg = error_msg + c + ','
    if error_msg != '':
        temp = list(error_msg)
        temp[-1] = ' '
        error_msg = ''.join(temp)
        raise Exception('there is Forbidden Character in {}. \n Forbidden Characters : {}'.format(
            error_msg, str(FORBIDDEN_CHARs)))

    if os.path.isfile(path):
        if clear_create:
            os.remove(path)
        else:
            print('{} is an existing file, do you want to replace? (y/n):'.format(path))
            command = ''
            while str.lower(command) != 'y' and str.lower(command) != 'n':
                command = input()
                if str.lower(command) == 'y':
                    os.remove(path)
                    print('{} REPLACED!'.format(path))

                elif str.lower(command) == 'n':
                    print('DATABASE CREATION ABORTED!')
                    return False
                else:
                    print('Wrong Command, please insert y for yes or n for no.')

    types = list(args)[-1]
    params_with_type = list(types.keys())

    file = open(path, 'w')

    first_line = ''
    if 'id' in columns:
        first_line = 'id:{}{}'.format(int, db_settings.splitter)
        columns.remove('id')

    for i, c in enumerate(columns):
        splitter = ''
        type = ''
        if i != len(columns)-1:
            splitter = db_settings.splitter
        if c in params_with_type:
            type = ':'+str(types[c])

        first_line = first_line + str.lower(c) + type + splitter

    file.write(first_line)
    file.close()

def get_db_info(path):
    if os.path.isfile(path):
        file = open(path, 'r')
        first_line = file.readline()
        raw_columns = first_line.split(db_settings.splitter)

        _column_names = []
        _column_types = {}
        for c in raw_columns:
            if ':' in c:
                temp = c.split(':')
                _column_names.append(temp[0].strip())

                _class_name = temp[1].replace("<class '", '').replace("'>", '')
                _class = eval(_class_name.strip())

                _column_types[temp[0].strip()] = _class
            elif c == 'id':
                _column_names.append(c.strip())
                _column_types[c.strip()] = int
            else:
                _column_names.append(c.strip())
                _column_types[c.strip()] = ''

        _has_id = 'id' in _column_names
        _col_count = len(_column_names)
        _row_count = len(file.readlines())
        return db_info(_column_names, _column_types, _col_count, _row_count, _has_id)
    else:
        raise Exception('db not found! (path : {})'.format(path))

def insert(path, data):
    """
    Args:
        path (string): path to existing database
        data (object or list of objects): data to save to database

    if input object has id it should be None for this method. this method generates id automaticly. for inserting with id you should use insertOrUpdate() function. 
    """

    if isinstance(data, list):
        _instance = data[0]
    else:
        _instance = data
        data = [data]

    attributes = inspect.getmembers(
        _instance, lambda a: not(inspect.isroutine(a)))
    columns = list(dict([a for a in attributes if not(
        a[0].startswith('__') and a[0].endswith('__'))]).keys())

    if not os.path.isfile(path):
        create(path, type(_instance))

    _db_info = get_db_info(path)
    
    temp_1 = columns.copy()
    temp_2 = _db_info.column_names.copy()
    temp_1.sort()
    temp_2.sort()
    if temp_1 != temp_2:
        raise Exception('db does not match with specified data!')

    if 'id' in columns:
        for c in data:
            if c.id != None:
                raise Exception(
                    "input object has id and it's values have to be None for this method. use insertOrUpdate if use want to set id values manually!")

    if not isinstance(data, list):
        data = [data]

    flag_rewrite_database = False
    if 'id' in columns:
        old_data = get_db(path, type(data[0]))
        if any(old_data):
            id_list_old = [d.id for d in old_data]
            id_list_old_set = set(id_list_old)
            if len(id_list_old_set) != len(id_list_old):
                command = get_command(
                    'Corrupted Database! id Conflicts found in database! clear them automaticly? (y/n):', ['y', 'n'])
                if command == 'y':
                    flag_rewrite_database = True
                    for id in id_list_old_set:
                        id_list_old = [d.id for d in old_data]
                        _count = len([i for i in id_list_old if i == id])
                        if _count > 1:
                            _conflicts = [d for d in old_data if d.id == id]
                            _max = max(id_list_old)
                            for i in range(_count-1):
                                _conflicts[-1-i].id = _max + 1
                                _max = _max + 1

                elif command == 'n':
                    raise Exception('Corrupted Database!')
        
        # feeding id to new data
        if any(old_data):
            id_list_old = [d.id for d in old_data]
            _max = max(id_list_old)
        else:
            _max = -1
        for i, c in enumerate(data):
            c.id = _max + 1
            _max = _max + 1

    if flag_rewrite_database:
        create(path, type(data[0]), True)
        data = old_data + data

    file = open(path, 'a')
    for obj in data:
        attributes = inspect.getmembers(
            obj, lambda a: not(inspect.isroutine(a)))
        row_data = dict([a for a in attributes if not(
            a[0].startswith('__') and a[0].endswith('__'))])

        _columns = _db_info.column_names.copy()
        if 'id' in columns:
            line = '\n{}{}'.format(str(row_data['id']), db_settings.splitter)
            _columns.remove('id')
        else:
            line = '\n'

        for i, c in enumerate(_columns):
            splitter = ''
            if i != len(_columns)-1:
                splitter = db_settings.splitter

            line = line + str(row_data[c]) + splitter
        file.write(line)
    file.close()

def insertOrUpdate(path, data):
    _db_info = get_db_info(path)
    if 'id' not in _db_info.column_names:
        raise Exception('insertOrUpdate is only for databases with id column!')

    if not isinstance(data, list):
        data = [data]

    old_data = get_db(path, type(data[0]))
    for new_obj in data:
        search = [d for d in old_data if d.id == new_obj.id]

        if len(search) > 1:
            print('Corrupted Database! multiple rows have same id!')
            print('do you want to clear conflicts? (y/n):')
            command = ''
            while str.lower(command) != 'y' and str.lower(command) != 'n':
                command = input()
                if command == 'y':
                    for i in range(len(search)-1):
                        old_data.remove(search[-1-i])
                        search.remove(search[-1-i])

                elif command == 'n':
                    raise Exception('Corrupted Database!')

        if len(search) == 1:
            _index = old_data.index(search[0])
            old_data[_index] = copy.copy(new_obj)
        else:
            old_data.append(new_obj)

    create(path, type(data[0]), clear_create=True)

    old_data.sort(key=lambda x: x.id)

    file = open(path, 'a')
    for obj in old_data:
        attributes = inspect.getmembers(
            obj, lambda a: not(inspect.isroutine(a)))
        row_data = dict([a for a in attributes if not(
            a[0].startswith('__') and a[0].endswith('__'))])

        line = '\n{}{}'.format(str(row_data['id']), db_settings.splitter)
        _columns = _db_info.column_names.copy()
        _columns.remove('id')

        for i, c in enumerate(_columns):
            splitter = ''
            if i != len(_columns)-1:
                splitter = db_settings.splitter

            line = line + str(row_data[c]) + splitter
        file.write(line)
    file.close()

def get_db(path, _class):
    if not os.path.isfile(path):
        raise Exception('db not found! (path : {})'.format(path))
    
    _db_info = get_db_info(path)

    # checking order of parameteres in the passed class
    args = inspect.getfullargspec(_class)
    columns = list(args)[0]
    if 'self' in columns:
        columns.remove('self')

    bad_order = False
    if _db_info.column_names != columns:
        for c in columns:
            if c not in _db_info.column_names:
                raise Exception('db does not match with specified class!')
        # if no exceptions -> order of parameters is different
        order_list = []
        for c in _db_info.column_names:
            order_list.append(columns.index(c))
        bad_order = True

    file = open(path, 'r')
    first_line = file.readline()
    raw_data = file.readlines()
    file.close()
    
    data = []
    for line in raw_data:
        raw_line_data = line.split(db_settings.splitter)
        line_data = []

        for i, c in enumerate(_db_info.column_names):
            if _db_info.column_types[c] != '':
                try:
                    if _db_info.column_types[c] == list:
                        line_data.append(json.loads(raw_line_data[i].strip()))
                    else:
                        line_data.append(_db_info.column_types[c](
                            raw_line_data[i].strip()))
                except:
                    raise(Exception(
                        "error occurred converting '{}' to class {}"
                        .format(raw_line_data[i].strip(), _db_info.column_types[c])))
            else:
                line_data.append(raw_line_data[i].strip())

        if bad_order:
            line_data = [x for _, x in sorted(zip(order_list, line_data))]
        _instanse = _class(*line_data)
        data.append(_instanse)

    return data

def remove_list(path, list : list):
    _db = get_db(path, type(list[0]))
    for instance in list:
        if instance in _db:
            _db.remove(instance)
    for i in range(len(_db)):
        _db[i].id = None
    
    create(path, type(list[0]), True)
    insert(path, _db)
    
def remove_id(path, id, _class):
    _db = get_db(path, _class)
    item = [i for i in _db if i.id == id]
    if len(item) == 0:
        raise(Exception('no item found with id = {}!'.format(str(id))))
    _db.remove(item[0])
    create(path, _class, True)
    for i in range(len(_db)):
        _db[i].id = None
    insert(path, _db)

def update_db_class(path, _class):
    if not os.path.isfile(path):
        raise Exception('db not found! (path : {})'.format(path))
    try:
        data = get_db(path, _class)
        raise Exception('passed Class matches the Database!')
    except:
        pass

    # Get DB Info
    _db_info = get_db_info(path)
    
    # Get Class Data
    args = inspect.getfullargspec(_class)
    all_columns = list(args)[0]
    if 'self' in all_columns:
        all_columns.remove('self')
        
    columns_with_type = list(args)[-1]

    
    # Reading Data
    file = open(path, 'r')
    _ = file.readline()
    raw_data = file.readlines()
    file.close()
    
            
            
    data = []
    for line in raw_data:
        raw_line_data = line.split(db_settings.splitter)
        line_data = []

        for i, c in enumerate(all_columns):
            # new column
            if c not in _db_info.column_names:
                column_value = None
            # existing column
            else:
                column_value = raw_line_data[_db_info.column_names.index(c)].strip()
                
            if c in columns_with_type.keys():
                if columns_with_type[c] == list:
                    column_value = json.loads(column_value)
                else:
                    columns_with_type[c](column_value)
            
            if column_value == None:
                column_value = ''
                        
            line_data.append(column_value)
    
        _instanse = _class(*line_data)
        data.append(_instanse)
    
    if 'id' in all_columns:
        for i, c in enumerate(range(len(data))):
            if c != None:
                data[i].id = None

    create(path, _class, True)
    insert(path, data)