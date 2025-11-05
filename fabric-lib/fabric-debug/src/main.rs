use std::{
    collections::HashSet,
    ffi::c_void,
    io::{Write, stdout},
    mem::MaybeUninit,
    ops::DerefMut,
    ptr::NonNull,
    sync::Arc,
    time::Instant,
};

use anyhow::anyhow;
use bytes::{BufMut, BytesMut};
use cuda_lib::{
    CudaDeviceId, CudaDeviceMemory, Device,
    cudart_sys::{cudaMemcpy, cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice},
    rt::cudaSetDevice,
};
use fabric_lib::{
    FabricEngine, RdmaDomainInfo, Worker,
    api::{
        DomainAddress, DomainGroupRouting, ImmTransferRequest, MemoryRegionDescriptor,
        MemoryRegionHandle, PagedTransferRequest, SingleTransferRequest,
        TransferCompletionEntry, TransferId, TransferRequest,
    },
    detect_topology,
};
use serde::{Deserialize, Serialize};

fn addr_to_string(addr: DomainAddress) -> String {
    let mut s = String::new();
    for b in &addr.0 {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn parse_addr(s: &str) -> anyhow::Result<DomainAddress> {
    if !s.len().is_multiple_of(2) {
        return Err(anyhow!("Expected even number of characters"));
    }
    let mut buf = BytesMut::with_capacity(s.len() / 2);
    for i in 0..s.len() / 2 {
        let byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)?;
        buf.put_u8(byte);
    }
    Ok(DomainAddress(buf.freeze()))
}

fn fill_random_u64(vec: &mut [u64], seed: u64) {
    let mut state = seed;
    for v in vec.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *v = state;
    }
}

fn fill_random_bytes(vec: &mut [u8], seed: u64) {
    assert!(vec.len().is_multiple_of(size_of::<u64>()));
    let vec64 = unsafe {
        std::slice::from_raw_parts_mut(
            vec.as_mut_ptr() as *mut u64,
            vec.len() / size_of::<u64>(),
        )
    };
    fill_random_u64(vec64, seed);
}

fn fill_paged(
    vec: &mut [u8],
    seed: u64,
    offset: usize,
    indices: &[u32],
    len: usize,
    stride: usize,
) {
    for (i, page) in indices.iter().enumerate() {
        let page = *page as usize;
        let off = offset + page * stride;
        assert!(off + len <= vec.len());
        let page_ptr = unsafe { vec.as_mut_ptr().add(off) };
        let page_slice = unsafe { std::slice::from_raw_parts_mut(page_ptr, len) };
        fill_random_bytes(page_slice, seed + i as u64);
    }
}

fn memcpy_h2d(dst: NonNull<c_void>, src: &[u8]) {
    assert_eq!(
        unsafe {
            cudaMemcpy(
                dst.as_ptr(),
                src.as_ptr() as *const c_void,
                src.len(),
                cudaMemcpyHostToDevice,
            )
        },
        0
    );
}

fn memcpy_d2h(dst: &mut [u8], src: NonNull<c_void>) {
    assert_eq!(
        unsafe {
            cudaMemcpy(
                dst.as_mut_ptr() as *mut c_void,
                src.as_ptr(),
                dst.len(),
                cudaMemcpyDeviceToHost,
            )
        },
        0
    );
}

macro_rules! wait_completion {
    ($engine:expr, $pattern:pat => $result:expr) => {{
        loop {
            std::hint::spin_loop();
            if let Some(comp) = $engine.poll_transfer_completion() {
                match comp {
                    $pattern => break Ok($result),
                    _ => break Err(anyhow!("Unexpected completion entry. {:?}", comp)),
                }
            }
        }
    }};
}

fn avg_std(list: &[f64]) -> (usize, f64, f64) {
    let n = list.len();
    let sum: f64 = list.iter().sum();
    let mean = sum / n as f64;
    let variance: f64 = list.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let stddev = variance.sqrt();
    (n, mean, stddev)
}

#[derive(Serialize, Deserialize)]
enum RequestContent {
    Paged(Paged),
    Single(Single),
    Imm(Imm),
}

#[derive(Serialize, Deserialize)]
struct RequestMeta {
    pub addr: DomainAddress,
    pub warmups: i32,
    pub repeats: i32,
}

#[derive(Serialize, Deserialize)]
struct Request {
    pub meta: RequestMeta,
    pub content: RequestContent,
}

#[derive(Serialize, Deserialize)]
struct Response {
    n: usize,
    elapsed_avg: f64,
    elapsed_std: f64,
    bw_avg: f64,
    bw_std: f64,
    bw_pct: f64,
    pps_avg: f64,
    pps_std: f64,
    pps_per_domain: f64,
}

#[derive(Clone, Serialize, Deserialize)]
struct Paged {
    pub seed: Option<u64>,
    pub mr_desc: Vec<MemoryRegionDescriptor>,
    pub offset: usize,
    pub indices: Vec<u32>,
    pub len: usize,
    pub stride: usize,
}

#[derive(Clone, Serialize, Deserialize)]
struct Single {
    pub seed: Option<u64>,
    pub mr_desc: MemoryRegionDescriptor,
    pub offset: usize,
    pub len: usize,
}

#[derive(Serialize, Deserialize)]
struct Imm {
    pub addr: DomainAddress,
    pub mr_desc: MemoryRegionDescriptor,
    pub imm: u32,
}

const MESSAGE_BUF_SIZE: usize = 64 << 20;
const CUDA_BUF_SIZE: usize = 256 << 20;

fn build_engine(args: &[String]) -> anyhow::Result<(Vec<u8>, FabricEngine)> {
    let topo_groups = detect_topology()?;
    let selected_gpus: Vec<_>;
    let nets_per_gpu;
    let all_gpus: Vec<_> = topo_groups.iter().map(|g| g.cuda_device).collect();
    if args.len() == 1 {
        selected_gpus = all_gpus.clone();
        nets_per_gpu = topo_groups[0].domains.len();
    } else {
        selected_gpus = args[1].split(',').map(|s| s.parse::<u8>().unwrap()).collect();
        nets_per_gpu = args[2].parse().unwrap();
    }

    // Prepare workers
    let mut workers = Vec::new();
    println!("Topology:");
    for &cuda_device in &selected_gpus {
        let topo_group = topo_groups
            .iter()
            .find(|g| g.cuda_device == cuda_device)
            .ok_or(anyhow!("Cannot find cuda:{} in topology", cuda_device))?;
        let domain_list: Vec<_> =
            topo_group.domains.iter().take(nets_per_gpu).cloned().collect();
        if domain_list.len() < nets_per_gpu {
            eprintln!("Not enough domains for GPU:{}", cuda_device);
            std::process::exit(1);
        }
        let worker_cpu = topo_group.cpus[0];
        let uvm_cpu = topo_group.cpus[1];

        print!(
            "  GPU: {}, NUMA: {}, Worker CPU: {:2}, UVM CPU: {:2}, NICs:",
            cuda_device, topo_group.numa, worker_cpu, uvm_cpu
        );
        for info in &domain_list {
            print!(" {:14}", info.name());
        }
        println!();
        let worker = Worker {
            domain_list,
            pin_worker_cpu: Some(worker_cpu),
            pin_uvm_cpu: Some(uvm_cpu),
        };
        workers.push((cuda_device, worker));
    }

    // Set up the fabric engine and its workers.
    let engine = FabricEngine::new(workers)?;

    Ok((selected_gpus, engine))
}

struct MemoryResource {
    _host_box: Box<MaybeUninit<[u8; MESSAGE_BUF_SIZE]>>,
    host_buf: NonNull<c_void>,
    host_mr_handle: MemoryRegionHandle,
    cuda_res: Vec<CudaResource>,
}

struct CudaResource {
    cuda_device: u8,
    cuda_buf: CudaDeviceMemory,
    cuda_mr_handle: MemoryRegionHandle,
    cuda_mr_desc: MemoryRegionDescriptor,
}

fn alloc_and_register_memory(
    selected_gpus: &[u8],
    engine: &FabricEngine,
) -> anyhow::Result<MemoryResource> {
    print!("Registering memory...");
    stdout().flush()?;
    let mut _host_box = Box::<[u8; MESSAGE_BUF_SIZE]>::new_uninit();
    let host_buf = unsafe {
        NonNull::new_unchecked(_host_box.deref_mut().as_mut_ptr() as *mut c_void)
    };
    let host_mr_handle =
        engine.register_memory_local(host_buf, MESSAGE_BUF_SIZE, Device::Host)?;
    print!(" cpu");
    stdout().flush()?;

    let mut cuda_res = Vec::new();
    for &cuda_device in selected_gpus {
        cudaSetDevice(cuda_device as i32)?;
        let cuda_buf = CudaDeviceMemory::device(CUDA_BUF_SIZE)?;
        let (cuda_mr_handle, cuda_mr_desc) = engine.register_memory_allow_remote(
            cuda_buf.ptr(),
            cuda_buf.size(),
            Device::Cuda(CudaDeviceId(cuda_device)),
        )?;
        cuda_res.push(CudaResource {
            cuda_device,
            cuda_buf,
            cuda_mr_handle,
            cuda_mr_desc,
        });
        print!(" cuda:{}", cuda_device);
        stdout().flush()?;
    }
    println!();

    Ok(MemoryResource { _host_box, host_buf, host_mr_handle, cuda_res })
}

fn unregister_memory(
    engine: &FabricEngine,
    host_buf: NonNull<c_void>,
    cuda_res: &[CudaResource],
) -> anyhow::Result<()> {
    engine.unregister_memory(host_buf)?;
    for res in cuda_res {
        engine.unregister_memory(res.cuda_buf.ptr())?;
    }
    Ok(())
}

fn generate_random_paged_data(request: &Paged, seed: u64, rank: usize) -> Vec<u8> {
    let mut tmp = vec![0u8; CUDA_BUF_SIZE];
    fill_paged(
        &mut tmp,
        seed << 4 | rank as u64,
        request.offset,
        request.indices.as_ref(),
        request.len,
        request.stride,
    );
    tmp
}

fn do_bench_write(
    engine: &FabricEngine,
    cuda_res: &[CudaResource],
    meta: &RequestMeta,
    request: &RequestContent,
) -> anyhow::Result<Response> {
    // Prepare transfer requests
    let total_ops;
    let total_bytes;
    let transfer_requests;
    let link_speed;
    match request {
        RequestContent::Paged(paged) => {
            // Generate random data if client wants to verify
            if let Some(seed) = paged.seed {
                for (i, res) in cuda_res.iter().enumerate() {
                    let tmp = generate_random_paged_data(paged, seed, i);
                    memcpy_h2d(res.cuda_buf.ptr(), &tmp);
                }
            }

            // Generate transfer requests
            total_ops = paged.indices.len() * cuda_res.len();
            total_bytes = paged.len * paged.indices.len() * cuda_res.len();
            link_speed = engine.aggregated_link_speed() as f64;
            let page_indices = Arc::new(paged.indices.clone());
            let mut reqs = Vec::new();
            for (i, res) in cuda_res.iter().enumerate() {
                let id = TransferId(1000 + res.cuda_device as u64);
                reqs.push((
                    i,
                    id,
                    TransferRequest::Paged(PagedTransferRequest {
                        src_page_indices: Arc::clone(&page_indices),
                        dst_page_indices: Arc::clone(&page_indices),
                        length: paged.len as u64,
                        src_mr: res.cuda_mr_handle,
                        src_stride: paged.stride as u64,
                        src_offset: paged.offset as u64,
                        dst_mr: paged.mr_desc[i].clone(),
                        dst_stride: paged.stride as u64,
                        dst_offset: paged.offset as u64,
                        imm_data: None,
                    }),
                ));
            }
            transfer_requests = reqs;
        }

        RequestContent::Single(single) => {
            let res = &cuda_res[0];

            // Generate random data if client wants to verify
            if let Some(seed) = single.seed {
                let mut tmp = vec![0u8; single.len];
                fill_random_bytes(&mut tmp, seed - 1);
                memcpy_h2d(unsafe { res.cuda_buf.ptr().byte_add(single.offset) }, &tmp);
            }

            // Generate transfer requests
            total_ops = 1;
            total_bytes = single.len;
            link_speed = engine.aggregated_link_speed() as f64 / cuda_res.len() as f64;
            let mut reqs = Vec::new();
            let id = TransferId(2000 + res.cuda_device as u64);
            reqs.push((
                0,
                id,
                TransferRequest::Single(SingleTransferRequest {
                    src_mr: res.cuda_mr_handle,
                    src_offset: single.offset as u64,
                    length: single.len as u64,
                    imm_data: None,
                    dst_mr: single.mr_desc.clone(),
                    dst_offset: single.offset as u64,
                    domain: DomainGroupRouting::RoundRobinSharded {
                        num_shards: engine.nets_per_gpu(),
                    },
                }),
            ));
            transfer_requests = reqs;
        }

        RequestContent::Imm(imm) => {
            total_ops = 1;
            total_bytes = 4;
            link_speed =
                engine.aggregated_link_speed() as f64 / engine.num_domains() as f64;
            transfer_requests = vec![(
                0,
                TransferId(3000),
                TransferRequest::Imm(ImmTransferRequest {
                    imm_data: imm.imm,
                    dst_mr: imm.mr_desc.clone(),
                    domain: DomainGroupRouting::Pinned { domain_idx: 0 },
                }),
            )]
        }
    }

    // Benchmark
    let mut elapsed_list = Vec::new();
    let mut bw_list = Vec::new();
    let mut pps_list = Vec::new();
    for repeat in -meta.warmups..meta.repeats {
        // Submit transfers
        let t0 = Instant::now();
        let mut pending_transfer_ids = HashSet::new();
        for (_i, id, req) in transfer_requests.iter() {
            pending_transfer_ids.insert(id);
            engine.submit_transfer(*id, req.clone(), None)?;
        }

        // Wait for transfer completion
        while !pending_transfer_ids.is_empty() {
            let transfer_id = wait_completion!(
                engine,
                TransferCompletionEntry::Transfer(transfer_id) => transfer_id
            )?;
            if !pending_transfer_ids.remove(&transfer_id) {
                return Err(anyhow!(
                    "Unexpected transfer completion. {:?}",
                    transfer_id
                ));
            }
        }
        let t1 = Instant::now();

        // Stats
        let elapsed = t1.duration_since(t0);
        let bw = total_bytes as f64 / elapsed.as_secs_f64() * 8.0;
        let pps = total_ops as f64 / elapsed.as_secs_f64();
        if repeat >= 0 {
            elapsed_list.push(elapsed.as_secs_f64());
            bw_list.push(bw);
            pps_list.push(pps);
        }
    }

    let (n, elapsed_avg, elapsed_std) = avg_std(&elapsed_list);
    let (_, bw_avg, bw_std) = avg_std(&bw_list);
    let (_, pps_avg, pps_std) = avg_std(&pps_list);
    Ok(Response {
        n,
        elapsed_avg,
        elapsed_std,
        bw_avg,
        bw_std,
        bw_pct: bw_avg / link_speed * 100.0,
        pps_avg,
        pps_std,
        pps_per_domain: pps_avg / engine.num_domains() as f64,
    })
}

fn server_main(args: Vec<String>) -> anyhow::Result<()> {
    if args.len() != 1 && args.len() != 3 {
        eprintln!("Server Usage:");
        eprintln!("Default runs with all GPUs and all NICs:");
        eprintln!("  {}", args[0]);
        eprintln!("  {}   selected_gpus     nets_per_gpu", args[0]);
        eprintln!("Example:");
        eprintln!("  {}   0,1,2,3,4,5,6,7   4", args[0]);
        std::process::exit(1);
    }

    let (selected_gpus, engine) = build_engine(&args)?;

    println!(
        "Initialized TransferEngine. num_nets={}, nets_per_gpu={}, link_speed={:.0}Gbps",
        engine.num_domains(),
        engine.num_domains() / selected_gpus.len(),
        engine.aggregated_link_speed() as f64 / 1e9,
    );
    println!("Main Address: {}", addr_to_string(engine.main_address()));

    let MemoryResource { _host_box, host_buf, host_mr_handle, cuda_res } =
        alloc_and_register_memory(&selected_gpus, &engine)?;

    loop {
        print!("Waiting for client to connect...");
        engine.submit_recv(
            TransferId(100),
            host_mr_handle,
            host_buf,
            MESSAGE_BUF_SIZE,
        )?;
        stdout().flush()?;
        let (_transfer_id, data_len) = wait_completion!(
            engine,
            TransferCompletionEntry::Recv { transfer_id, data_len } => (transfer_id, data_len)
        )?;

        // Deserialize request
        let data = unsafe {
            std::slice::from_raw_parts(host_buf.as_ptr() as *const u8, data_len)
        };
        let request: Request = postcard::from_bytes(data)?;

        // Print message
        let msg = match &request.content {
            RequestContent::Paged(paged) => {
                format!(
                    "Paged: page_bytes={}, num_pages={}",
                    paged.len,
                    paged.indices.len()
                )
            }
            RequestContent::Single(single) => {
                format!("Single: bytes={}", single.len)
            }
            RequestContent::Imm(_) => "Imm".to_string(),
        };
        print!("\rRequest: {} ...\x1b[K", msg);
        stdout().flush()?;

        // Do benchmark
        let response =
            do_bench_write(&engine, &cuda_res, &request.meta, &request.content)?;
        println!(" Done!");

        // Send response
        let data = postcard::to_slice(&response, unsafe {
            std::slice::from_raw_parts_mut(
                host_buf.as_ptr() as *mut u8,
                MESSAGE_BUF_SIZE,
            )
        })?;
        engine.submit_send(
            TransferId(200),
            request.meta.addr,
            host_mr_handle,
            host_buf,
            data.len(),
        )?;
        wait_completion!(
            engine,
            TransferCompletionEntry::Send(_) => ()
        )?;
    }
}

fn generate_paged_write_request(
    cuda_res: &[CudaResource],
    page_bytes: usize,
    num_pages: usize,
    verify: bool,
) -> Paged {
    let stride = page_bytes + 128;
    let offset = 1024;
    let max_pages = (CUDA_BUF_SIZE - offset) / stride;
    let seed = if verify { Some(0xabcdabcd987u64) } else { None };

    // Generate page_indices
    let mut rng_u64 = vec![0u64; num_pages];
    fill_random_u64(&mut rng_u64, 0x123);
    let page_indices: Vec<_> =
        rng_u64.into_iter().map(|x| (x as usize % max_pages) as u32).collect();

    Paged {
        seed,
        mr_desc: cuda_res.iter().map(|res| res.cuda_mr_desc.clone()).collect(),
        offset,
        indices: page_indices.clone(),
        len: page_bytes,
        stride,
    }
}

fn verify_paged_write(cuda_res: &[CudaResource], paged: &Paged) {
    if let Some(seed) = paged.seed {
        for (i, res) in cuda_res.iter().enumerate() {
            let gold = generate_random_paged_data(paged, seed, i);
            let mut buf = vec![0u8; CUDA_BUF_SIZE];
            memcpy_d2h(&mut buf, res.cuda_buf.ptr());
            for &page_idx in paged.indices.iter() {
                let offset = paged.offset + page_idx as usize * paged.stride;
                let page_gold = &gold[offset..offset + paged.len];
                let page_buf = &buf[offset..offset + paged.len];
                assert!(page_gold == page_buf);
            }
        }
    }
}

fn generate_single_write_request(
    cuda_res: &[CudaResource],
    write_bytes: usize,
    verify: bool,
) -> Single {
    let offset = 1024;
    let seed = if verify { Some(0xabcdabcd987u64) } else { None };

    Single { seed, mr_desc: cuda_res[0].cuda_mr_desc.clone(), offset, len: write_bytes }
}

fn verify_single_write(cuda_res: &[CudaResource], single: &Single) {
    if let Some(seed) = single.seed {
        let res = &cuda_res[0];
        let mut gold = vec![0u8; single.len];
        fill_random_bytes(&mut gold, seed - 1);
        let mut tmp = vec![0u8; single.len];
        memcpy_d2h(&mut tmp, unsafe { res.cuda_buf.ptr().byte_add(single.offset) });
        assert!(gold == tmp);
    }
}

fn do_request(
    engine: &FabricEngine,
    host_buf: NonNull<c_void>,
    host_mr_handle: MemoryRegionHandle,
    cuda_res: &[CudaResource],
    server_addr: DomainAddress,
    request_content: RequestContent,
) -> anyhow::Result<Response> {
    // Send request
    let request = Request {
        meta: RequestMeta { addr: engine.main_address(), warmups: 50, repeats: 100 },
        content: request_content,
    };
    let data = postcard::to_slice(&request, unsafe {
        std::slice::from_raw_parts_mut(host_buf.as_ptr() as *mut u8, MESSAGE_BUF_SIZE)
    })?;
    engine.submit_send(
        TransferId(100),
        server_addr,
        host_mr_handle,
        host_buf,
        data.len(),
    )?;

    // Submit RECV
    engine.submit_recv(TransferId(200), host_mr_handle, host_buf, MESSAGE_BUF_SIZE)?;

    // Wait for SEND completion
    wait_completion!(
        engine,
        TransferCompletionEntry::Send(_) => ()
    )?;

    // Wait for Imm
    if let RequestContent::Imm(imm_req) = &request.content {
        for _ in -request.meta.warmups..request.meta.repeats {
            let imm_data = wait_completion!(
                engine,
                TransferCompletionEntry::ImmData(imm_data) => imm_data
            )?;
            assert_eq!(imm_data, imm_req.imm);
        }
    }

    // Wait for RECV completion
    let (_, data_len) = wait_completion!(
        engine,
        TransferCompletionEntry::Recv { transfer_id, data_len } => (transfer_id, data_len)
    )?;

    // Deserialize response
    let data =
        unsafe { std::slice::from_raw_parts(host_buf.as_ptr() as *const u8, data_len) };
    let response: Response = postcard::from_bytes(data)?;

    // Verify data
    match &request.content {
        RequestContent::Paged(paged) => {
            verify_paged_write(cuda_res, paged);
        }
        RequestContent::Single(single) => {
            verify_single_write(cuda_res, single);
        }
        RequestContent::Imm(_) => {}
    }

    Ok(response)
}

fn client_main(args: Vec<String>) -> anyhow::Result<()> {
    if args.len() != 4 {
        eprintln!("Client Usage:");
        eprintln!("  {} selected_gpus nets_per_gpu server_address", args[0]);
        std::process::exit(1);
    }
    let server_addr = parse_addr(&args[3])?;

    let (selected_gpus, engine) = build_engine(&args)?;
    println!(
        "Initialized TransferEngine. num_nets={}, nets_per_gpu={}, link_speed={:.0}Gbps",
        engine.num_domains(),
        engine.num_domains() / selected_gpus.len(),
        engine.aggregated_link_speed() as f64 / 1e9,
    );

    let MemoryResource { _host_box, host_buf, host_mr_handle, cuda_res } =
        alloc_and_register_memory(&selected_gpus, &engine)?;

    println!("Bench Paged Write");
    for (page_bytes, num_pages, verify) in [
        (10000, 200, true),
        (1024, 10000, false),
        (4096, 10000, false),
        (8192, 10000, false),
        (16384, 10000, false),
        (32768, 10000, false),
        (65536, 10000, false),
        (1024, 100, false),
        (4096, 100, false),
        (8192, 100, false),
        (16384, 100, false),
        (32768, 100, false),
        (65536, 100, false),
    ] {
        print!("  page_bytes: {:5}, num_pages: {:5} ...", page_bytes, num_pages);
        stdout().flush()?;
        let content = RequestContent::Paged(generate_paged_write_request(
            &cuda_res, page_bytes, num_pages, verify,
        ));
        let r = do_request(
            &engine,
            host_buf,
            host_mr_handle,
            &cuda_res,
            server_addr.clone(),
            content,
        )?;
        if verify {
            println!(" VERIFIED");
            continue;
        }
        print!(" lat: {:6.3} ± {:6.3} ms", r.elapsed_avg * 1e3, r.elapsed_std * 1e3);
        print!(", bw: {:4.0} ± {:4.0} Gbps", r.bw_avg / 1e9, r.bw_std / 1e9);
        print!(" ({:2.0}%)", r.bw_pct);
        print!(", rate: {:6.3} ± {:5.3} Mpps", r.pps_avg / 1e6, r.pps_std / 1e6,);
        print!(" (per domain: {:6.3} Mpps)", r.pps_per_domain / 1e6);
        println!();
    }

    println!("Bench Single Write");
    for (write_bytes, verify) in [
        (10000, true),
        (1 << 10, false),
        (2 << 10, false),
        (4 << 10, false),
        (8 << 10, false),
        (16 << 10, false),
        (32 << 10, false),
        (64 << 10, false),
        (128 << 10, false),
        (256 << 10, false),
        (512 << 10, false),
        (1 << 20, false),
        (2 << 20, false),
        (4 << 20, false),
        (8 << 20, false),
        (16 << 20, false),
        (32 << 20, false),
        (64 << 20, false),
    ] {
        print!("  bytes: {:8} ...", write_bytes);
        stdout().flush()?;
        let content = RequestContent::Single(generate_single_write_request(
            &cuda_res,
            write_bytes,
            verify,
        ));
        let r = do_request(
            &engine,
            host_buf,
            host_mr_handle,
            &cuda_res,
            server_addr.clone(),
            content,
        )?;
        if verify {
            println!(" VERIFIED");
            continue;
        }
        print!(" lat: {:6.3} ± {:6.3} ms", r.elapsed_avg * 1e3, r.elapsed_std * 1e3);
        print!(", bw: {:4.0} ± {:4.0} Gbps", r.bw_avg / 1e9, r.bw_std / 1e9);
        print!(" ({:2.0}%)", r.bw_pct);
        println!();
    }

    println!("Bench Imm Write");
    {
        let imm = 20250501;
        let content = RequestContent::Imm(Imm {
            addr: engine.main_address(),
            mr_desc: cuda_res[0].cuda_mr_desc.clone(),
            imm,
        });
        let r = do_request(
            &engine,
            host_buf,
            host_mr_handle,
            &cuda_res,
            server_addr.clone(),
            content,
        )?;
        print!("  Imm VERIFIED ... ");
        print!(" lat: {:3.3} ± {:3.3} µs", r.elapsed_avg * 1e6, r.elapsed_std * 1e6);
        println!();
    }

    println!("Done!");

    unregister_memory(&engine, host_buf, &cuda_res)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    logging_lib::init(&logging_lib::LoggingOpts {
        log_format: logging_lib::LogFormat::Text,
        log_color: logging_lib::LogColor::Auto,
        log_directives: None,
    })?;
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() <= 3 { server_main(args) } else { client_main(args) }
}
