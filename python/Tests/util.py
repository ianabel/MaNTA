import toml
import sys

def get_transport_system_as_module(config_path):
    filename = config_path

    Config = toml.load( filename )

    conf = Config.get("configuration")
    if conf == None:
        print("Missing [configuration] section in " + filename)
        sys.exit(1)

    import importlib.util
    import time

    if "PythonModuleName" in conf:
        mod_name = conf["PythonModuleName"]
        if "PythonModuleFile" in conf:
            mod_path = conf["PythonModuleFile"]
        else:
            print("Must specify a file containing a python module if you specify a python module to use")
            sys.exit(1)
        mod_spec = importlib.util.spec_from_file_location( mod_name, mod_path )
        module_handle = importlib.util.module_from_spec( mod_spec )
        sys.modules[ mod_name ] = module_handle
        try:
            mod_spec.loader.exec_module(module_handle)
        except Exception:
            # If execution fails, remove the module from sys.modules
            del sys.modules[mod_name]
            raise

        print(f"Using python module {mod_name} from file {mod_path}\n")

        module_handle.registerTransportSystems()

        return filename