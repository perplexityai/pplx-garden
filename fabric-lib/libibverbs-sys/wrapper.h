#include <infiniband/verbs.h>

static inline int ibv_query_port_wrap(
    struct ibv_context *context,
    uint8_t port_num,
    struct ibv_port_attr *port_attr)
{
    // ibv_query_port is a macro. Have to use this trick to make bindgen work.
    return ibv_query_port(context, port_num, port_attr);
}
