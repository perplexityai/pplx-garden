use std::ptr::NonNull;

use crate::{
    error::{Result, VerbsError},
    verbs::verbs_address::{Gid, VerbsRCAddress, VerbsUDAddress},
};
use libibverbs_sys::{
    IBV_ACCESS_LOCAL_WRITE, IBV_ACCESS_REMOTE_READ, IBV_ACCESS_REMOTE_WRITE,
    IBV_QP_ACCESS_FLAGS, IBV_QP_AV, IBV_QP_DEST_QPN, IBV_QP_MAX_DEST_RD_ATOMIC,
    IBV_QP_MAX_QP_RD_ATOMIC, IBV_QP_MIN_RNR_TIMER, IBV_QP_PATH_MTU, IBV_QP_PKEY_INDEX,
    IBV_QP_PORT, IBV_QP_QKEY, IBV_QP_RETRY_CNT, IBV_QP_RNR_RETRY, IBV_QP_RQ_PSN,
    IBV_QP_SQ_PSN, IBV_QP_STATE, IBV_QP_TIMEOUT, IBV_QPS_INIT, IBV_QPS_RTR,
    IBV_QPS_RTS, IBV_QPT_RC, IBV_QPT_UD, ibv_cq, ibv_create_qp, ibv_destroy_qp,
    ibv_modify_qp, ibv_mtu, ibv_pd, ibv_qp, ibv_qp_attr, ibv_qp_cap, ibv_qp_init_attr,
    ibv_srq,
};

pub struct UDQueuePair {
    pub addr: VerbsUDAddress,
    pub qp: NonNull<ibv_qp>,
}

impl UDQueuePair {
    pub fn new(
        cq: NonNull<ibv_cq>,
        pd: NonNull<ibv_pd>,
        gid: Gid,
        lid: u16,
        qkey: u32,
        max_wr: u32,
    ) -> Result<Self> {
        let mut qp_init_attr = ibv_qp_init_attr {
            qp_type: IBV_QPT_UD,
            sq_sig_all: 0,
            send_cq: cq.as_ptr(),
            recv_cq: cq.as_ptr(),
            cap: ibv_qp_cap {
                max_send_wr: max_wr,
                max_recv_wr: max_wr,
                max_send_sge: 1,
                max_recv_sge: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let qp =
            NonNull::new(unsafe { ibv_create_qp(pd.as_ptr(), &raw mut qp_init_attr) })
                .ok_or_else(|| VerbsError::with_last_os_error("ibv_create_qp: UD"))?;
        let qp_num = unsafe { (*qp.as_ptr()).qp_num };
        Ok(UDQueuePair { addr: VerbsUDAddress { gid, lid, qp_num, qkey }, qp })
    }

    pub fn ud_reset_to_init(&self, pkey_index: u16, port_num: u8) -> Result<()> {
        let mut qp_attr = ibv_qp_attr {
            qp_state: IBV_QPS_INIT,
            pkey_index,
            port_num,
            qkey: self.addr.qkey,
            ..Default::default()
        };
        let flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY;
        let ec =
            unsafe { ibv_modify_qp(self.qp.as_ptr(), &raw mut qp_attr, flags as i32) };
        if ec != 0 {
            return Err(
                VerbsError::with_code(ec, "ibv_modify_qp: ud_reset_to_init").into()
            );
        }
        Ok(())
    }

    pub fn ud_init_to_rtr(&self) -> Result<()> {
        let mut qp_attr = ibv_qp_attr { qp_state: IBV_QPS_RTR, ..Default::default() };
        let flags = IBV_QP_STATE;
        let ec =
            unsafe { ibv_modify_qp(self.qp.as_ptr(), &raw mut qp_attr, flags as i32) };
        if ec != 0 {
            return Err(
                VerbsError::with_code(ec, "ibv_modify_qp: ud_init_to_rtr").into()
            );
        }
        Ok(())
    }

    pub fn ud_rtr_to_rts(&self) -> Result<()> {
        let mut qp_attr = ibv_qp_attr { qp_state: IBV_QPS_RTS, ..Default::default() };
        // UD doesn't need sq_psn but verbs API requires it.
        let flags = IBV_QP_STATE | IBV_QP_SQ_PSN;
        let ec =
            unsafe { ibv_modify_qp(self.qp.as_ptr(), &raw mut qp_attr, flags as i32) };
        if ec != 0 {
            return Err(
                VerbsError::with_code(ec, "ibv_modify_qp: ud_rtr_to_rts").into()
            );
        }
        Ok(())
    }

    pub fn destroy(&mut self) {
        unsafe {
            ibv_destroy_qp(self.qp.as_ptr());
        }
    }
}

pub struct RCQueuePair {
    // TODO: state enum
    pub addr: VerbsRCAddress,
    pub qp: NonNull<ibv_qp>,
}

impl RCQueuePair {
    pub fn new(
        cq: NonNull<ibv_cq>,
        pd: NonNull<ibv_pd>,
        srq: NonNull<ibv_srq>,
        gid: Gid,
        lid: u16,
        max_wr: u32,
        psn: u32,
    ) -> Result<Self> {
        let mut qp_init_attr = ibv_qp_init_attr {
            qp_type: IBV_QPT_RC,
            sq_sig_all: 0,
            send_cq: cq.as_ptr(),
            recv_cq: cq.as_ptr(),
            srq: srq.as_ptr(),
            cap: ibv_qp_cap {
                max_send_wr: max_wr,
                max_send_sge: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let qp =
            NonNull::new(unsafe { ibv_create_qp(pd.as_ptr(), &raw mut qp_init_attr) })
                .ok_or_else(|| VerbsError::with_last_os_error("ibv_create_qp"))?;
        let qp_num = unsafe { (*qp.as_ptr()).qp_num };
        Ok(RCQueuePair { addr: VerbsRCAddress { gid, lid, qp_num, psn }, qp })
    }

    pub fn rc_reset_to_init(&self, port_num: u8, pkey_index: u16) -> Result<()> {
        let mut qp_attr = ibv_qp_attr {
            qp_state: IBV_QPS_INIT,
            pkey_index,
            port_num,
            qp_access_flags: IBV_ACCESS_LOCAL_WRITE
                | IBV_ACCESS_REMOTE_READ
                | IBV_ACCESS_REMOTE_WRITE,
            ..Default::default()
        };
        let flags =
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
        let ec =
            unsafe { ibv_modify_qp(self.qp.as_ptr(), &raw mut qp_attr, flags as i32) };
        if ec != 0 {
            return Err(
                VerbsError::with_code(ec, "ibv_modify_qp: rc_reset_to_init").into()
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rc_init_to_rtr(
        &self,
        is_infiniband: bool,
        gid_index: u8,
        port_num: u8,
        path_mtu: ibv_mtu,
        peer_qp_num: u32,
        peer_psn: u32,
        peer_lid: u16,
        peer_gid: Gid,
    ) -> Result<()> {
        let mut qp_attr = ibv_qp_attr {
            qp_state: IBV_QPS_RTR,
            path_mtu,
            dest_qp_num: peer_qp_num,
            rq_psn: peer_psn,
            max_dest_rd_atomic: 1,
            min_rnr_timer: 12,
            ..Default::default()
        };
        qp_attr.ah_attr.port_num = port_num;
        qp_attr.ah_attr.sl = 0;
        if is_infiniband {
            qp_attr.ah_attr.is_global = 0;
            qp_attr.ah_attr.dlid = peer_lid;
        } else {
            qp_attr.ah_attr.is_global = 1;
            qp_attr.ah_attr.grh.dgid = peer_gid.into();
            qp_attr.ah_attr.grh.hop_limit = 64;
            qp_attr.ah_attr.grh.sgid_index = gid_index;
        }
        let flags = IBV_QP_STATE
            | IBV_QP_AV
            | IBV_QP_PATH_MTU
            | IBV_QP_DEST_QPN
            | IBV_QP_RQ_PSN
            | IBV_QP_MAX_DEST_RD_ATOMIC
            | IBV_QP_MIN_RNR_TIMER;
        let ec =
            unsafe { ibv_modify_qp(self.qp.as_ptr(), &raw mut qp_attr, flags as i32) };
        if ec != 0 {
            return Err(
                VerbsError::with_code(ec, "ibv_modify_qp: rc_init_to_rtr").into()
            );
        }
        Ok(())
    }

    pub fn rc_rtr_to_rts(&self, peer_psn: u32) -> Result<()> {
        let mut qp_attr = ibv_qp_attr {
            qp_state: IBV_QPS_RTS,
            timeout: 14,
            retry_cnt: 7,
            rnr_retry: 7,
            sq_psn: peer_psn,
            max_rd_atomic: 1,
            ..Default::default()
        };
        let flags = IBV_QP_STATE
            | IBV_QP_TIMEOUT
            | IBV_QP_RETRY_CNT
            | IBV_QP_RNR_RETRY
            | IBV_QP_SQ_PSN
            | IBV_QP_MAX_QP_RD_ATOMIC;
        let ec =
            unsafe { ibv_modify_qp(self.qp.as_ptr(), &raw mut qp_attr, flags as i32) };
        if ec != 0 {
            return Err(
                VerbsError::with_code(ec, "ibv_modify_qp: rc_rtr_to_rts").into()
            );
        }
        Ok(())
    }

    pub fn destroy(&mut self) {
        unsafe {
            ibv_destroy_qp(self.qp.as_ptr());
        }
    }
}
