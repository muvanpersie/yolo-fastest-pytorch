import tensorrt as trt
import cv2
import os

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class RTInfer():
    def __init__(self, model_path, input_height, input_width):
        self.input_height = input_height
        self.input_width = input_width

        self.model_path = model_path
        self.use_dynamic = False
    
    def build_enigne(self):
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        TRT_LOGGER = trt.Logger()
        
        if self.model_path.endswith('trt'):
            with open(self.model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        else:
            with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config:
                builder.max_batch_size = 1
                config.max_workspace_size = 1 << 30

                network = builder.create_network(EXPLICIT_BATCH)
                parser = trt.OnnxParser(network, TRT_LOGGER)

                if self.use_dynamic:
                    profile = builder.create_optimization_profile()
                    profile.set_shape("input", (1, 3, 64, 64), (1, 3, 512, 1024), (1, 3, 960, 1920)) 
                    config.add_optimization_profile(profile)

                with open(self.model_path, 'rb') as model:
                    if not parser.parse(model.read()):
                        print ('ERROR: Failed to parse the ONNX file.')

                self.engine = builder.build_engine(network, config)

                save_engine_path = self.model_path.replace("onnx", "trt")
                with open(save_engine_path, "wb") as f:
                    f.write(self.engine.serialize())

    
    def infer(self, input_img):
        with self.engine.create_execution_context() as context:
            outputs = []
            inputs, outputs, bindings = self.allocate_buffers()

            inputs[0].host = input_img
            stream=cuda.Stream()

            [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

            # context.set_binding_shape(0, (1, 3, 768, 1088))
            context.set_binding_shape(0, (1, 3, self.input_height, self.input_width))

            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            
            [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

            stream.synchronize()

            return [out.host for out in outputs]

            # return outputs
    

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []

        HEIGHT = self.input_height
        WIDTH = self.input_width

        for binding in self.engine: # binding 'input'...'reg_2d' 
            
            ##### dynamic  ############
            # if self.engine.binding_is_input(binding):
            #     size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size * HEIGHT * WIDTH
            # else:
            #     size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size * int(HEIGHT / 8) * int(WIDTH / 8)

            
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings #, stream
