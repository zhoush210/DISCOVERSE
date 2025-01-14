import time
import grpc
import numpy as np
import array_pb2
import array_pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = array_pb2_grpc.ArrayServiceStub(channel)

    stt = time.time()
    sec = 1.
    fqt = 1000
    nct = int(fqt * sec)
    for i in range(nct):
        get_observation = np.array(stub.GetArray(array_pb2.GetArrayResponse()).data, dtype=np.float32)
        sim_time = get_observation[0]
        jq = np.array(np.array2string(get_observation[1:8], separator=", "))
        jv = np.array(np.array2string(get_observation[8:15], separator=", "))
        jf = np.array(np.array2string(get_observation[15:22], separator=", "))
        print("time:", sim_time)
        print("jq:", jq)
        print("jv:", jv)
        print("jf:", jf)

        action_array = 2. * (np.random.random(7) - 0.5)
        set_request = array_pb2.SetArrayRequest(data=action_array.tolist())

        set_response = stub.SetArray(set_request)
        print("Set method success:", set_response.success)

        ct = time.time()
        time.sleep(max(1./fqt - (ct - stt - i/fqt), 0.))

    edt = time.time()
    print("Real time: ", np.allclose(edt - stt, sec, atol=2e-3), f" at frequency: {fqt}")

if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True, linewidth=500)
    run()