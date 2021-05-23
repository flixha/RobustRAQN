# modified from https://gist.github.com/schlamar/2311116#file-processify-py-L17
# also see http://stackoverflow.com/questions/2046603/is-it-possible-to-run-function-in-a-subprocess-without-threading-or-writing-a-se

# %%
from datetime import time
import inspect
import os
import sys
import traceback

from functools import wraps

from multiprocessing import Process, Queue, SimpleQueue
import queue


class Sentinel:
    pass


def processify(func):
    '''Decorator to run a function as a process.
    Be sure that every argument and the return value
    is *pickable*.
    The created process is joined, so the code does not
    run in parallel.
    '''

    def process_generator_func(q, *args, **kwargs):
        result = None
        error = None
        it = iter(func())
        while error is None and result != Sentinel:
            try:
                result = next(it)
                error = None
            except StopIteration:
                result = Sentinel
                error = None
            except Exception:
                ex_type, ex_value, tb = sys.exc_info()
                error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
                result = None
            q.put((result, error))

    def process_func(q, *args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception:
            ex_type, ex_value, tb = sys.exc_info()
            error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
            result = None
        else:
            error = None

        q.put((result, error))

    def wrap_func(*args, **kwargs):
        # register original function with different name
        # in sys.modules so it is pickable
        process_func.__name__ = func.__name__ + 'processify_func'
        setattr(sys.modules[__name__], process_func.__name__, process_func)

        q = Queue()
        p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
        p.start()
        try:
            result, error = q.get()
        except queue.Empty:
            result = None
            error = None
        p.join()
        # p.terminate()

        if error:
            ex_type, ex_value, tb_str = error
            # try:
            message = '%s (in subprocess)\n%s' % (ex_value.message, tb_str)
            # except Exception:
            #     message = '%s (in subprocess)\n%s' % (ex_value.message, tb_str)
            try:
                exception = ex_type(message)
            except Exception:
                # Failed to keep the original exception type
                exception = Exception('%s\n(original exception type: %s)'
                                      % (message, ex_type))
            raise exception

        return result

    def wrap_generator_func(*args, **kwargs):
        # register original function with different name
        # in sys.modules so it is pickable
        process_generator_func.__name__ = func.__name__ + 'processify_generator_func'
        setattr(sys.modules[__name__], process_generator_func.__name__,
                process_generator_func)

        q = Queue()
        p = Process(target=process_generator_func, args=[q] + list(args),
                    kwargs=kwargs)
        p.start()

        result = None
        error = None
        while error is None:
            result, error = q.get()
            if result == Sentinel:
                break
            yield result
        p.join()
        #while not <myevent>.is_set():
        #    wait()
        # p.terminate()

        if error:
            ex_type, ex_value, tb_str = error
            message = '%s (in subprocess)\n%s' % (ex_value.message, tb_str)
            try:
                exception = ex_type(message)
            except Exception:
                # Failed to keep the original exception type
                exception = Exception('%s\n(original exception type: %s)'
                                      % (message, ex_type))
            raise exception

    @wraps(func)
    def wrapper(*args, **kwargs):
        if inspect.isgeneratorfunction(func):
            return wrap_generator_func(*args, **kwargs)
        else:
            return wrap_func(*args, **kwargs)
    return wrapper


@processify
def test_function():
    return os.getpid()


@processify
def test_generator_func():
    for msg in ["generator", "function"]:
        yield msg


@processify
def test_deadlock():
    return range(30000)


@processify
def test_exception():
    raise RuntimeError('xyz')


def test():
    print(os.getpid())
    print(test_function())
    print(list(test_generator_func()))
    print(len(test_deadlock()))
    test_exception()

if __name__ == '__main__':
    test()

# %%
# import os
# import sys
# import traceback
# from functools import wraps
# from multiprocessing import Process, Queue


# def processify(func):
#     '''Decorator to run a function as a process.
#     Be sure that every argument and the return value
#     is *pickable*.
#     The created process is joined, so the code does not
#     run in parallel.
#     '''

#     def process_func(q, *args, **kwargs):
#         try:
#             ret = func(*args, **kwargs)
#         except Exception:
#             ex_type, ex_value, tb = sys.exc_info()
#             error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
#             ret = None
#         else:
#             error = None

#         q.put((ret, error))

#     # register original function with different name
#     # in sys.modules so it is pickable
#     process_func.__name__ = func.__name__ + 'processify_func'
#     setattr(sys.modules[__name__], process_func.__name__, process_func)

#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         q = Queue()
#         p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
#         p.start()
#         ret, error = q.get()

#         if error:
#             ex_type, ex_value, tb_str = error
#             message = '%s (in subprocess)\n%s' % (ex_value.message, tb_str)
#             raise ex_type(message)

#         return ret
#     return wrapper