#!/usr/bin/env python
"""
vLLM API 서버 런처 스크립트
각 GPU마다 독립적인 vLLM 서버를 실행
"""
import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
import signal
import json

def launch_vllm_server(gpu_id: int, port: int, config_path: str):
    """단일 GPU에서 vLLM 서버 실행"""
    
    # 설정 파일 로드
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_name = config['model']['base_model']
    vllm_config = config['data']['raw_dataset']['vllm']
    
    # 환경 변수 설정
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # 로그 레벨 제어 (INFO/DEBUG/WARNING)
    env.setdefault('VLLM_LOG_LEVEL', env.get('VLLM_LOG_LEVEL', 'INFO'))
    
    # vLLM 서버 실행 명령
    cmd = [
        sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', model_name,
        '--port', str(port),
        '--host', '0.0.0.0',
        '--gpu-memory-utilization', str(vllm_config['gpu_memory_utilization']),
        '--max-model-len', str(vllm_config.get('max_model_len', 32768)),
        '--dtype', vllm_config['dtype'],
        '--generation-config', 'vllm',
        '--max-num-batched-tokens', str(vllm_config.get('max_num_batched_tokens', 16384)),
        '--max-num-seqs', str(vllm_config.get('max_num_seqs', 256)),
        '--disable-log-stats' if vllm_config.get('disable_log_stats', False) else '',
        '--trust-remote-code' if vllm_config.get('trust_remote_code', False) else '',
        '--enable-prefix-caching',  # 프리픽스 캐싱 활성화
        # 요청 로그는 기본 활성 (필요시 VLLM_LOG_LEVEL로 제어)
    ]
    
    # KV 캐시 dtype 설정
    if 'kv_cache_dtype' in vllm_config:
        cmd.extend(['--kv-cache-dtype', vllm_config['kv_cache_dtype']])
    
    # 빈 문자열 제거
    cmd = [c for c in cmd if c]
    
    print(f"🚀 GPU {gpu_id}에서 vLLM 서버 시작 (포트: {port})")
    print(f"명령어: {' '.join(cmd)}")

    # 서버 로그 디렉토리 준비
    workspace = os.environ.get('WORKSPACE', '/mnt/data1/projects/Conf_Agg')
    log_dir = Path(os.environ.get('SERVER_LOG_DIR', str(Path(workspace) / 'outputs/logs/vllm')))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'server_gpu_{gpu_id}.log'
    print(f"로그 파일: {log_path}")
    
    # 서버 프로세스 시작
    log_file = open(log_path, 'a')
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=log_file,
        text=True
    )
    
    return process

def wait_for_server(port: int, max_retries: int = 30):
    """서버가 준비될 때까지 대기"""
    import requests
    
    url = f"http://localhost:{port}/health"
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ 포트 {port}의 서버가 준비되었습니다.")
                return True
        except:
            pass
        
        print(f"서버 준비 대기 중... ({i+1}/{max_retries})")
        time.sleep(5)
    
    return False

def main():
    parser = argparse.ArgumentParser(description="vLLM API 서버 런처")
    parser.add_argument('--num-gpus', type=int, default=4, help='사용할 GPU 수')
    parser.add_argument('--base-port', type=int, default=8000, help='시작 포트 번호')
    parser.add_argument('--config-path', type=str, required=True, help='설정 파일 경로')
    args = parser.parse_args()
    
    processes = []
    ports = []
    
    try:
        # 각 GPU마다 서버 실행
        for gpu_id in range(args.num_gpus):
            port = args.base_port + gpu_id
            process = launch_vllm_server(gpu_id, port, args.config_path)
            processes.append(process)
            ports.append(port)
            
            # 서버가 준비될 때까지 대기
            if not wait_for_server(port):
                print(f"❌ GPU {gpu_id}의 서버 시작 실패")
                raise Exception(f"서버 시작 실패: GPU {gpu_id}")
        
        print("\n" + "="*50)
        print("✅ 모든 vLLM 서버가 성공적으로 시작되었습니다!")
        print(f"포트: {ports}")
        print("="*50 + "\n")
        
        # 서버 정보를 파일로 저장 (WORKSPACE 기준)
        server_info = {
            'servers': [
                {'gpu_id': i, 'port': ports[i], 'url': f'http://localhost:{ports[i]}'}
                for i in range(args.num_gpus)
            ]
        }
        workspace = os.environ.get('WORKSPACE', '/mnt/data1/projects/Conf_Agg')
        out_path = Path(workspace) / 'vllm_servers.json'
        with open(out_path, 'w') as f:
            json.dump(server_info, f, indent=2)
        print(f"서버 정보를 저장했습니다: {out_path}")
        print("서버를 종료하려면 Ctrl+C를 누르세요...")
        
        # 프로세스 모니터링: 재시작 없이 종료 감지 시 즉시 에러 반환
        while True:
            for i, process in enumerate(processes):
                ret = process.poll()
                if ret is not None:
                    print(f"❌ GPU {i}의 서버가 종료되었습니다. 재시작하지 않습니다. (return code: {ret})")
                    sys.exit(1)
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n서버 종료 중...")
        for process in processes:
            process.terminate()
            process.wait()
        print("모든 서버가 종료되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
        for process in processes:
            process.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()