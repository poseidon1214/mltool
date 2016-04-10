#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
#   datemark: 20150428
#   
#   V1.1:
#       add MailSender and Arg
#
#   V1.0.6 change::
#       add xfind
#       xfind: set operation. treat file as set.
#
#   V1.0.5 change::
#       add VarConf and RandomItemGenerator
#
#   V1.0.4 change::
#       add topkheap from zhangduo@
#
#   V1.0.3 change::
#       add Timer.
#
#   V1.0.2 change::
#       add Mapper mode. (--mapper)
#
#   V1.0.1 change:: 
#       dump(self, stream, sort)
#
#   V1.0
#       complete code.
# 


import os
import re
import logging
import traceback
import socket
import sys
import time
from multiprocessing import *
import heapq
import itertools
import random
import ConfigParser
import argparse
import json

#import threading


HEADER_LENGTH = 8
DETECTIVE_MSG = 'Are_you_alive?'

##############################################################################
# Part I: pydev library implemention.
#
##############################################################################

class MailSender:
    def __init__(self, sendmail='/usr/sbin/sendmail'):
        self.__cmd = sendmail

    def send(self, receiver, title, cont, sender='pydev.MailSender@nickgu.github.com'):
        #p = os.popen('cat', 'w')
        p = os.popen(self.__cmd + ' %s'%receiver, 'w')
        print >> p, 'From: pydev.MailSender<%s>' % sender
        print >> p, 'Sender: pydev.MailSender<%s>' % sender
        print >> p, 'To: %s' % ','.join(map(lambda x:'%s'%(x), receiver.split(' ')))
        print >> p, 'Subject: %s\n' % title
        print >> p, cont
        p.close()

class RandomItemGenerator:
    '''
    input a item stream and then output N random item.
    '''
    def __init__(self, N):
        self.__random_num = N
        self.__ol = []
        self.__nth = 0

    def feed(self, item):
        if len(self.__ol)<self.__random_num:
            self.__ol.append( item )
        else:
            x = random.randint(0, self.__nth)
            if x < self.__random_num:
                self.__ol[x] = item
        self.__nth += 1

    def result(self):
        return self.__ol

# from zhangduo.
class TopkHeap(object):
    def __init__(self, k, key_func):
        self.k = k
        self.key_func = key_func
        self.data = []
        self.counter = itertools.count() # unique sequence count

    def get_data(self):
        return [x[2] for x in self.data]
    
    def sorted_data(self):
        return [x[2] for x in reversed([heapq.heappop(self.data) for x in xrange(len(self.data))])]

    def extend_heap(self, size):
        self.k += size

    def push(self, elem):
        if len(self.data) < self.k:
            count = next(self.counter)
            heapq.heappush(self.data, [self.key_func(elem), count, elem])
            return True
        else:
            small_key, _, _ = self.data[0]
            elem_key = self.key_func(elem)
            if elem_key > small_key:
                count = next(self.counter)
                heapq.heapreplace(self.data, [elem_key, count, elem])
                return True
        return False

class VarConfig:
    def __init__(self):
        self.__config = {}
        self.__config_context = {}

    def read(self, filenames, var_opt=None, var_sec='var', conf_template='conf_template'):
        ''' 
            use var_opt(dict) and var_section to load default param.
            which will subtitute %(param)s in config.

            Step 1: load all raw conf.
                generate [(section.option, raw_info)] tuple.

            Step 2: if conf_template is set:
                makeup each conf by it's template conf.
                conf template relation must be DAG.

            Step 3: 
                for section in DAG-order:
                subtitute the params.
        '''
        raw_conf = ConfigParser.ConfigParser()
        raw_conf.read(filenames)
        # step 1. read row.
        dependency = {}
        section_list = list(raw_conf.sections())
        if conf_template:
            raw_conf = ConfigParser.ConfigParser()
            raw_conf.read(filenames)
            for section in raw_conf.sections():
                if raw_conf.has_option(section, conf_template):
                    dependency[section] = raw_conf.get(section, conf_template, raw=True)
        # generate DAG-order.
        dag_section_list = []
        for section in section_list:
            temp = []
            while 1:
                if section in self.__config:
                    break
                temp.insert(0, section)
                self.__config[section] = {}
                if section in dependency:
                    section = dependency[section]
                else:
                    break
            dag_section_list += temp

        # step 2. 
        #   makeup config by dependency.
        for section in dag_section_list:
            for opt in raw_conf.options(section):
                value = raw_conf.get(section, opt, raw=True)
                self.__config[section][opt] = value
            self.__makeup_config(section, dependency.get(section, None))

        # step 3.
        #   substitute params.
        default_var = {}
        if var_sec:
            default_var = self.__config.get(var_sec, {})
        for section in dag_section_list:
            self.__config_context[section] = self.__overwrite_dict([default_var, self.__config[section], var_opt])

    def get(self, sec, opt, default=None, raw=False, throw_exception=True):
        if sec in self.__config:
            if opt in self.__config[sec]:
                if raw:
                    return self.__config[sec][opt]
                return self.__interpolate(
                            self.__config[sec][opt], 
                            self.__config_context[sec], 
                            throw_exception=throw_exception)
        return default

    def has_section(self, sec):
        return sec in self.__config

    def has_option(self, sec, opt):
        if sec in self.__config:
            if opt in self.__config[sec]:
                return True
        return False

    def items(self, section, raw=False, throw_exception=True):
        for key, value in self.__config.get(section, {}).iteritems():
            if raw:
                yield key, value
            else:
                yield key, self.__interpolate(value, self.__config_context[section], throw_exception)

    def options(self, section):
        return self.__config.get(section, {}).keys()

    def sections(self):
        return self.__config.keys()

    def clear(self):
        self.__config = {}

    def __overwrite_dict(self, dict_list):
        ret = {}
        for d in dict_list:
            if isinstance(d, dict):
                for key, value in d.iteritems():
                    ret[key] = value
        return ret
        
    def __makeup_config(self, son, father):
        if father is None:
            return 
        for key in self.__config[father].keys():
            if key not in self.__config[son]:
                self.__config[son][key] = self.__config[father][key]

    def __interpolate(self, rawval, vars, throw_exception=True):
        ''' code from ConfigParser()
        '''
        # do the string interpolation
        value = rawval
        depth = self.MAX_INTERPOLATION_DEPTH
        while depth:                    # Loop through this until it's done
            depth -= 1
            if value and "%(" in value:
                value = self._KEYCRE.sub(self._interpolation_replace, value)
                try:
                    value = value % vars
                except KeyError, e:
                    if throw_exception:
                        raise Exception(('InterpolationMissingOptionError v=[%s]\n'
                                '\tbad value=%s\n'
                                '\tvals=%s\n') 
                                % (value, e, vars))
            else:
                break
        if value and "%(" in value:
            if throw_exception:
                raise Exception('InterpolationDepthErro')
            else:
                logging.error('VarConf: interpolation: [%s] failed.', rawval)
        return value

    MAX_INTERPOLATION_DEPTH = 32
    _KEYCRE = re.compile(r"%\(([^)]*)\)s|.")
    def _interpolation_replace(self, match):
        s = match.group(1)
        if s is None:
            return match.group()
        else:
            return "%%(%s)s" % s.lower()

class Arg(object):
    '''
    Sample code:
        ag=pydev.Arg()
        ag.str_opt('f', 'file', 'this arg is for file')
        opt = ag.init_arg()
        # todo with opt, such as opt.file
    '''
    def __init__(self, help='Lazy guy, no help'):
        self.is_parsed = False;
        help = help.decode('utf-8').encode('gb18030')
        self.__parser = argparse.ArgumentParser(description=help)
        self.__args = None;
        #    -l --log 
        self.str_opt('log', 'l', 'logging level default=[error]', meta='[debug|info|error]');
    def __default_tip(self, default_value=None):
        if default_value==None:
            return ''
        return ' default=[%s]'%default_value

    def bool_opt(self, name, iname, help=''):
        help = help.decode('utf-8').encode('gb18030')
        self.__parser.add_argument(
                '-'+iname, 
                '--'+name, 
                action='store_const', 
                const=1,
                default=0,
                help=help);
        return

    def str_opt(self, name, iname, help='', default=None, meta=None):
        help = (help + self.__default_tip(default)).decode('utf-8').encode('gb18030')
        self.__parser.add_argument(
                '-'+iname, 
                '--'+name, 
                metavar=meta,
                help=help,
                default=default);
        pass

    def var_opt(self, name, meta='', help='', default=None):
        help = (help + self.__default_tip(default).decode('utf-8').encode('gb18030'))
        if meta=='':
            meta=name
        self.__parser.add_argument(name,
                metavar=meta,
                help=help,
                default=default) 
        pass

    def init_arg(self, input_args=None):
        if not self.is_parsed:
            if input_args is not None:
                self.__args = self.__parser.parse_args(input_args)
            else:
                self.__args = self.__parser.parse_args()
            self.is_parsed = True;
        if self.__args.log:
            format='%(asctime)s %(levelname)8s [%(filename)18s:%(lineno)04d]: %(message)s'
            if self.__args.log=='debug':
                logging.basicConfig(level=logging.DEBUG, format=format)
                logging.debug('log level set to [%s]'%(self.__args.log));
            elif self.__args.log=='info':
                logging.basicConfig(level=logging.INFO, format=format)
                logging.info('log level set to [%s]'%(self.__args.log));
            elif self.__args.log=='error':
                logging.basicConfig(level=logging.ERROR, format=format)
                logging.info('log level set to [%s]'%(self.__args.log));
            else:
                logging.error('log mode invalid! [%s]'%self.__args.log)
        return self.__args

    @property
    def args(self):
        if not self.is_parsed:
            self.__args = self.__parser.parse_args()
            self.is_parsed = True;
        return self.__args;


def foreach_line(fd=sys.stdin, percentage=False):
    if percentage:
        cur_pos = fd.tell()
        fd.seek(0, 2)
        file_size = fd.tell()
        fd.seek(cur_pos)
        old_perc = 0
    while 1:
        line = fd.readline()
        if line == '':
            break
        if percentage:
            cur_pos = fd.tell()
            perc = int(100.0 * cur_pos / file_size)
            if perc>old_perc:
                old_perc = perc
                print >> sys.stderr, '%c[foreach_line] process %d%% (%d/%d)' % (
                        13, perc, cur_pos, file_size)
        yield line.strip('\n')


def foreach_row(fd=sys.stdin, min_fields_num=-1, seperator='\t', percentage=False):
    if percentage:
        cur_pos = fd.tell()
        fd.seek(0, 2)
        file_size = fd.tell()
        fd.seek(cur_pos)
        old_perc = 0
    while 1:
        line = fd.readline()
        if line == '':
            break
        if percentage:
            cur_pos = fd.tell()
            perc = int(100.0 * cur_pos / file_size)
            if perc>old_perc:
                old_perc = perc
                print >> sys.stderr, '%c[foreach_line] process %d%% (%d/%d)' % (
                        13, perc, cur_pos, file_size)
        arr = line.strip('\n').split(seperator)
        if min_fields_num>0 and len(arr)<min_fields_num:
            continue
        yield arr

def dict_from_str(s, l1_sep=';', l2_sep='='):
    dct = {}
    if not isinstance(s, str):
        return {}
    for item in s.split(l1_sep):
        r = item.split(l2_sep)
        if len(r)!=2:
            logging.error('[dict_from_str]: [%s] is not a valid item.')
            continue
        dct[r[0]] = r[1]
    return dct


def dict_from_file(fd=sys.stdin, process=None, key_process=None, seperator='\t'):
    dct = {}
    for row in foreach_row(fd, seperator=seperator):
        if key_process:
            key = key_process(row)
        else:
            key = row[0]

        if process:
            value = process(row)
        else:
            value = '\t'.join(row[1:])
        dct[key] = value
    return dct

def echo(input_text):
    return ('ACK: ' + input_text)

def sock_recv(sock):
    d = sock.recv(HEADER_LENGTH)
    if len(d)==0:
        return None
    data_len = int(d)
    #print data_len
    data = ''
    while 1:
        n = min(4096, data_len)
        d = sock.recv(n)
        if not d:
            break

        data_len -= len(d)
        data += d

        #print 'left=%d cur=%d' % (data_len, len(data))
        if data_len<=0:
            break
    return data

def sock_send(sock, data):
    data_len = '%8d' % len(data)
    sock.sendall(data_len)
    sock.sendall(data)

def simple_query(query, ip='127.0.0.1', port=12345):
    sys.stderr.write('SEEK_TO: %s:%s\n' % (ip, port))
    clisock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clisock.connect((ip, port))

    sock_send(clisock, query)
    ret = sock_recv(clisock)
    clisock.close()
    return ret

def detect(ip='127.0.0.1', port=12345):
    try:
        ret = simple_query(DETECTIVE_MSG, ip, port)
        if ret != 'YES':
            return False
    except Exception, msg:
        sys.stderr.write('detect err: %s\n' % msg)
        return False
    return True

def simple_query_by_name(query, name, ip='127.0.0.1'):
    cmd = 'SEEK\t%s' % name
    ret = simple_query(cmd, ip, port=8769)
    arr = ret.split('\t')
    if arr[0] != 'OK':
        sys.stderr.write('seek name failed! [%s]' % ret)
        return None
    port = int(arr[1])
    return simple_query(query, ip, port)

class BasicService:
    def __init__(self):
        self.__handler_init = None
        self.__handler_process = None
        self.__handler_timer_process = None
        self.__timer = 0.0

    def set_init(self, h_init):
        self.__handler_init = h_init

    def set_process(self, h_process):
        self.__handler_process = h_process

    def set_timer_deamon(self, h_process, seconds=60.0):
        '''
            set a process which will be called each time interval.
        '''
        self.__handler_timer_process = h_process
        self.__timer = seconds

    def run_with_name(self, name, desc='No description.', ip='127.0.0.1', port=12345):
        cmd = 'REGEDIT\t%s\t%d\t%s' % (name, port, desc)
        ret = simple_query(cmd, ip, port=8769)
        arr = ret.split('\t')
        if arr[0] != 'OK': 
            sys.stderr.write('SET NAME FAILED! [%s]' % ret)
            return
        self.run(ip, port)
        
    def run(self, ip='127.0.0.1', port=12345):
        if self.__handler_init:
            sys.stderr.write('init..\n')
            self.__handler_init()

        self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1) 
        self.__sock.bind( (ip, port) )
        self.__sock.listen(32)
        sys.stderr.write('listen : %s:%d\n' % (ip, port))

        last_time = time.time()
        
        try:
            while 1:
                # check time at first.
                if self.__handler_timer_process:
                    dt = time.time() - last_time
                    if dt >= self.__timer:
                        try:
                            self.__handler_timer_process()
                        except Exception, msg:
                            sys.stderr.write('error in time_handler: %s\n' % msg)
                        last_time = time.time()

                # set a timer for accept:
                #   because i need to run a timer process.
                self.__sock.settimeout(1);
                try:
                    clisock, (remote_host, remote_port) = self.__sock.accept()
                except socket.timeout, msg:
                    continue

                try:
                    data = sock_recv(clisock)
                    if data == DETECTIVE_MSG:
                        sock_send(clisock, 'YES')
                    else:
                        sys.stderr.write('[%s:%s] connected dl=%d\n' % (remote_host, remote_port, len(data)))
                        if self.__handler_process:
                            response = self.__handler_process(data)
                            if response:
                                sock_send(clisock, response)

                except Exception, msg:
                    sys.stderr.write('err [%s]!\n' % msg)
                    traceback.print_stack()
                    traceback.print_exc()
                    continue
                finally:
                    clisock.close()
        finally:
            sys.stderr.write('byebye.\n')
            self.__sock.close()

class ManagerService:
    def __init__(self):
        self.__name_dct = {}
        self.__desc_dct = {}
        self.__recover()
        self.__svr = BasicService()
        self.__svr.set_process(self.process)
        self.__svr.set_timer_deamon(self.deamon_process, 5)

    def run(self):
        self.__svr.run(port=8769)

    def deamon_process(self):
        '''
            check whether each service is alive.
        '''
        sys.stderr.write('detect: %s\n' % time.asctime()) 
        del_names = []
        for name, port in self.__name_dct.iteritems():
            alive = detect(port=port)
            if not alive:
                sys.stderr.write('%s : %s[%d] is dead.\n' % (time.asctime(), name, port))
                del_names.append(name)
        for name in del_names:
            del self.__name_dct[name]
            del self.__desc_dct[name]
        self.__backup()

    def process(self, cmd):
        '''
            3 type(s) of cmd:
                'SEEK[\t][name]' => 'OK\tPORT' or 'ERR\tNOT_FOUND'
                'REGEDIT[\t][name][\t][port][\t][desc]' => 'OK' or 'ERR\tmsg'
                'INFO' => 'OK\tname info.'
        '''
        cmd = cmd.replace('\n', '')
        cmd = cmd.replace('###', '')
        cmd = cmd.replace('||', '')
        arr = cmd.split('\t')
        if arr[0] == 'SEEK':
            if len(arr)!=2:
                return 'ERR\tpara_num=%d' % len(arr)
            name = arr[1]
            if name not in self.__name_dct:
                return 'ERR\tNOT_FOUND'
            return 'OK\t%d' % self.__name_dct[name] 
        elif arr[0] == 'REGEDIT':
            if len(arr)!=4:
                return 'ERR\tpara_num=%d' % len(arr)
            name, port, desc = arr[1:4]
            if ':' in name:
                return 'ERR\tINVALID_NAME_NO_:_'
            port = int(port)
            self.__name_dct[name] = port
            self.__desc_dct[name] = desc
            return 'OK'
        elif arr[0] == 'INFO':
            info = ''
            for name, port in self.__name_dct.iteritems():
                desc = self.__desc_dct.get(name, '')
                info += '%s||%s||%s###' % (name, port, desc)
            return 'OK\t%s' % info
    
    def __recover(self):
        try:
            f = file('service_info.txt')
        except:
            sys.stderr.write('no backup info.\n')
            return
        for l in f.readlines():
            arr = l.strip('\n').split('\t')
            if len(arr)!=3: 
                continue
            name, port, desc = arr
            port = int(port)
            if name not in self.__name_dct:
                self.__name_dct[name] = port
                self.__desc_dct[name] = desc

    def __backup(self):
        f = file('service_info.txt', 'w')
        for name, port in self.__name_dct.iteritems():
            desc = ''
            if name in self.__desc_dct:
                desc = self.__desc_dct[name]
            f.write('%s\t%d\t%s\n' % (name, port, name))
        f.close()

class CounterObject:
    def __init__(self):
        self.__keyvalue = dict()
        self.__keyvalue['__tag__'] = []

    def tag(self, t):
        self.__keyvalue['__tag__'].append(t)

    def kv(self, key, value):
        if key not in self.__keyvalue:
            self.__keyvalue[key] = []
        self.__keyvalue[key].append(value)

    def __str__(self):
        return json.dumps(self.__keyvalue)

    def loads(self, s):
        self.__keyvalue = json.loads(s)

class MapperCounter:
    def __init__(self):
        self.__dct = {}

    def inc(self, key, inc=1):
        if key not in self.__dct:
            self.__dct[key] = 0
        self.__dct[key] += inc

    def dump(self, stream, sort=False):
        if sort:
            for key, value in sorted(self.__dct.iteritems(), key=lambda x:-x[1]):
                print '%s\t%s' % (key, value)
        else:
            for key, value in self.__dct.iteritems():
                print '%s\t%s' % (key, value)

def __test_basic_service():
    # test a svr.
    svr = BasicService()
    svr.set_process(echo)
    svr.run_with_name('ECHO', desc='This is a echo service.')

class MPProcessor:
    def __init__(self, functor, proc_num, stdout_dir='mp_out'):
        self.functor = functor
        self.proc_num = proc_num
        self.processes = [];
        self.stdout_dir=stdout_dir;
        self.stdout_fn = [];
        for i in range(proc_num-1):
            self.processes.append(Process(target=self._inner_func, args=(i, )));
            out_fn = './%s/part-%05d'%(self.stdout_dir, i)
            self.stdout_fn.append(out_fn);
        # 补充一个thread num的FN
        out_fn = './%s/part-%05d'%(self.stdout_dir, self.proc_num)
        self.stdout_fn.append(out_fn);
        return

    def _inner_func(self, cur_i):
        old_stdout = sys.stdout;
        out_fn = self.stdout_fn[cur_i];
        logging.info('Process[%d] reset stdout to %s'%(cur_i, out_fn));
        sys.stdout = open( out_fn, 'w' )

        # 开始正式执行程序
        logging.info('Process[%d] begin to process.'%cur_i);
        # 执行进程函数、给定i和文件
        self.functor(cur_i, self.proc_num);

        sys.stdout = old_stdout; # 恢复bak stdout.
        logging.info('Process[%d] processes over.'%cur_i);

    def process_all(self):
        ''' START => JOIN.
        '''
        for process in self.processes:
            process.start();
    
        # 自己也跑一个。
        self._inner_func(self.proc_num-1);

        for process in self.processes:
            process.join();

class Timer:
    def __init__(self):
        self.clear()

    def begin(self):
        self.__begin_time = time.time()

    def end(self):
        self.__end_time = time.time()
        self.__total_time += self.cost_time()
        self.__counter += 1

    def cost_time(self):
        return self.__end_time - self.__begin_time

    def total_time(self):
        return self.__total_time

    def qps(self):
        qps = self.__counter / self.__total_time
        return qps

    def clear(self):
        self.__begin_time = None
        self.__end_time = None
        self.__counter = 0
        self.__total_time = 0
       
    def log(self, stream=sys.stderr, name=None, output_qps=False):
        qps_info = ''
        if output_qps:
            qps_info = 'QPS=%.3f' % (self.qps())
        if name:
            print >> stream, '[Timer][%s]: %.3f(s) %s' % (name, self.cost_time(), qps_info)
        else:
            print >> stream, '[Timer]: %.3f(s) %s' % (self.cost_time(), qps_info)

class MTItemProcessor(MPProcessor):
    def __init__(self, 
            proc_set, functor, proc_num, stdout_dir):
        MPProcessor.__init__(functor, proc_num, stdout_dir);
        self.proc_set = proc_set
        self.inner_func = functor
        self.functor = self._shell_functor
        return ;
    
    def _shell_functor(self, cur_i):
        for it in self.proc_set:
            if (id/7) % (self.proc_num+1) == cur_i:
                # hit this processor.
                self.inner_func(it);

    def merge_stdout(self):
        logging.info('MTP: merge stdout');
        line_cnt = 0;
        for fn in self.stdout_fn:
            fl = open(fn, 'r');
            line = fl.readline();
            while line:
                line=line.rstrip('\n');
                print line;
                line_cnt += 1;
                line = fl.readline()
            fl.close();
        logging.info('MTP: merge over! %d lines written.'%line_cnt);

##############################################################################
# Part II: CMD definition.
#  How to add a CMD:
#  def CMD_xx:
#   ''' doc.
#   '''
#   # your code.
#
#  xx will be command name.
#  doc will be the help doc as cmd.
#
##############################################################################

def CMD_random(argv):
    '''Generate random lines from stdin.
        Params:
            random [random_num]
    '''
    random_num = 10
    if len(argv)>0:
        random_num = int(argv[1])
    print >> sys.stderr, 'Random_num = %d' % random_num
    rd = RandomItemGenerator(random_num)
    for line in foreach_line():
        rd.feed(line)
    for item in rd.result():
        print item

def CMD_mgrservice(argv):
    '''Run the basic_service manager.
    '''
    s = ManagerService()
    s.run()

def CMD_show(argv):
    '''Show all the commands.
    '''
    l = sys.modules['__main__'].__dict__.keys()
    for key in l:
        if key.find('CMD_') == 0:
            print ' %s: ' % key.replace('CMD_', '')
            f = eval(key)
            if f.__doc__ is None:
                print '    [NO_DOC]'
                print
            else:
                print '    %s' % (f.__doc__.replace('\n', '\n    '))

def CMD_counter(argv):
    '''Run counter job.
        -i          : output int.
        --mapper    : run as mapper mode.
        -c [int]    : cut threshold.
    '''
    output_int = False
    arg_set = set(argv)
    cut_num = 0
    mapper_mode = False
    if '-i' in arg_set:
        # output as integer.
        output_int = True
    if '--mapper' in arg_set:
        mapper_mode = True
    for arg in arg_set:
        if arg.find('-c') == 0:
            cut_num = int(arg[2:])

    if mapper_mode:
        ct = MapperCounter()
        while 1:
            line = sys.stdin.readline()
            if line == '':
                break
            ct.inc(line.strip('\n'))
        ct.dump(sys.stdout)

    else:
        # reducer.
        last_key = None
        acc_value = 0
        while 1:
            line = sys.stdin.readline()
            if line == '':
                break
            arr = line.strip('\n').split('\t')
            if len(arr)!=2:
                continue
            key, value = arr
            if output_int:
                value = int(value)
            else:
                value = float(value)
            if key != last_key:
                if last_key:
                    if acc_value >= cut_num:
                        print '%s\t%s' % (last_key, acc_value)
                last_key = key
                acc_value = 0
            acc_value += value
        if last_key:
            if acc_value >= cut_num:
                print '%s\t%s' % (last_key, acc_value)

def CMD_test_conf(argv):
    cp = VarConfig()
    cp.read(argv, throw_exception=False)
    for sec in cp.sections():
        print '[%s]' % sec
        options = cp.options(sec)
        for k in options:
            try:
                v = cp.get(sec, k)
                print '%s.%s=%s' % (sec, k, v)
            except:
                print '%s.%s [error]' % (sec, k)
                continue
        print 

def CMD_xfind(argv):
    '''
    xfind: do set-operation in files.
        load dict from file(B), read data from stdin(A), make set-operation(find, A_B)
        xfind -f filename [-h] [-o <opeartion>] [-a --field_A] [-b --field_B] [-s --seperator <'\\t'>]
    '''
    a = Arg('load dict from file(B), read data from stdin(A), make set-operation(find, A_B)')
    a.str_opt('filename', 'f', 'input key file as [B]')
    a.str_opt('operation', 'o', 'operations, support find(A in B), A_B(A minus B)', default='find')
    a.str_opt('field_A', 'a', 'which row of A(input stream) will be treated as key, start at 1', default='1') 
    a.str_opt('field_B', 'b', 'which row of B(key_file) will be treated as key, start at 1', default='1') 
    a.str_opt('seperator', 's', 'seperator, default is tab.', default='\t') 
    opt = a.init_arg(argv)

    field_B = int(opt.field_B) - 1
    field_A = int(opt.field_A) - 1
    keydict = dict_from_file(file(opt.filename), process=lambda x:'\t'.join(x), key_process=lambda x:x[field_B], seperator=opt.seperator)
    logging.info('Dict loaded. size=%d' % len(keydict))
    for row in foreach_row(sys.stdin, seperator=opt.seperator):
        if len(row)<=field_A:
            continue
        if opt.operation == 'find':
            if row[field_A] in keydict:
                print opt.seperator.join(row)
        elif opt.operation == 'A_B':
            if row[field_A] not in keydict:
                print opt.seperator.join(row)

def CMD_sendmail(argv):
    '''
    send a file as mail to somebody.
    sendmail <receiver> <filename> <title>
    '''
    if len(argv)!=3:
        print 'sendmail <receiver> <filename> <title>'
        return 
    receiver, filename, title = argv
    s = MailSender()

    content = ''.join(file(filename).readlines())
    s.send(receiver, title, content)

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    '''
    pydev.py <command>
        command-list:
            list:
                list all the availble command.
    '''
    if len(sys.argv)<=1:
        print (
'''Usage:
    pydev.py <command>
    you can use 'pydev.py show' to get all available command.
''')
        sys.exit(-1)

    com = eval('CMD_' + sys.argv[1])
    ret = com(sys.argv[2:])
    sys.exit(ret)




