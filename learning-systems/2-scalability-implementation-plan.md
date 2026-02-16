# Scalability Implementation Plan: Fitness AI Assistant

This document outlines how to apply scalability concepts from `1-scalability-notes.md` to the fitness-ai-assistant project.

## Current Architecture

```
User Query (Gradio UI or REST API)
    ↓
[Inference Pipeline]
    ├─ Load embeddings (numpy arrays)
    ├─ Embed query (SentenceTransformer)
    ├─ Search vector DB (Milvus Lite)
    └─ Return top-k results with metadata
```

**Current bottlenecks:**
- Embedding computation on every query (CPU/GPU bound)
- Vector DB search for every unique query
- No caching of popular queries
- Single-node deployment (no scaling)
- No replication for vector DB

---

## Implementation Ideas

### 1. Caching for Similarity Search

**Goal**: Reduce repeated embedding computations and vector searches.

**Approach: Cache-Aside Pattern**

```
Query → Check Cache
    ├─ Hit: Return cached results
    └─ Miss: Embed → Search → Store in cache → Return
```

**Implementation Details:**

- **Cache store**: Redis or in-memory (e.g., `functools.lru_cache` for MVP)
- **Cache key**: hash(query_text + embedding_model + top_k)
- **TTL**: 15-60 minutes (depends on freshness requirements)
- **Eviction**: LRU (Least Recently Used) when memory full

**Benefits:**
- Reduces embedding computation by ~70-80% for common queries
- Reduces vector search load on Milvus
- Lower latency for cached queries (sub-100ms)

**Trade-offs:**
- Memory overhead (estimate: 1-5GB for 10K cached queries)
- Stale results for updated embeddings
- Cache invalidation complexity

**Implementation Plan:**
1. Add Redis or simple in-memory cache to `InferencePipeline`
2. Wrap search method with cache decorator
3. Add cache invalidation endpoint (admin only)
4. Monitor cache hit rate

**Code location**: `src/inference_pipeline/pipeline.py` (add cache layer)

---

### 2. Load Balancer (Multi-Client Deployment)

**Goal**: Handle multiple concurrent clients querying the same service.

**Current state**: Single instance of Gradio/REST API running on one port.

**Scaled deployment:**

```
        ┌─────────────┐
Clients→│  LB (nginx) │
        └──────┬──────┘
           ┌───┼───┐
           ↓   ↓   ↓
      [API-1][API-2][API-3]
           └───┬───┘
               ↓
        [Shared Vector DB]
        [Shared Metadata]
        [Shared Cache]
```

**Load Balancer Options:**

1. **Nginx** (recommended for MVP)
   - Lightweight L7 reverse proxy
   - Round-robin or least-connections balancing
   - Fast (~10k req/sec per core)

2. **HAProxy**
   - Powerful TCP/HTTP load balancer
   - Better health checks and failover

3. **Cloud LB** (AWS ALB, GCP LB)
   - Managed service
   - Auto-scaling friendly

**Sticky Sessions Consideration:**

Since search results are stateless (each query is independent), **sticky sessions are NOT required**. However, the cache must be **shared** across all instances.

**Shared Cache Strategy:**
- Use **Redis** instead of in-memory cache
- All API instances connect to same Redis
- Cache key: same across instances
- Hit rate improves with shared pool

**Implementation Plan:**
1. Switch from in-memory to Redis cache
2. Create Nginx config with upstream app servers
3. Deploy multiple instances of `src/inference_pipeline/gradio_app.py`
4. Add health check endpoint

**Config location**: New file `infra/nginx.conf`

---

### 3. Vector DB Scaling & Replication

**Goal**: Ensure vector DB survives node failures and scales to larger datasets.

**Current state**: Milvus Lite (single SQLite backend, no replication).

**Problems with current setup:**
- No redundancy (data loss if file corrupted)
- Single point of failure
- Limited to disk size of one machine
- No horizontal scaling

**Scaling Options:**

#### Option A: Milvus Standalone → Milvus Distributed

Upgrade from Milvus Lite to Milvus cluster:

```
[Milvus Master]
    ├─ Query Node 1
    ├─ Query Node 2
    └─ Query Node 3

+ Index Node
+ Data Node
+ Coordinator Nodes
```

**Pros:**
- Horizontal scaling for reads (multiple query nodes)
- Built-in replication
- High availability

**Cons:**
- More infrastructure (requires coordinator, separate storage)
- More operational complexity
- Overkill if dataset fits on one machine

#### Option B: Single Milvus + PostgreSQL Replication

Keep Milvus single-node for vector search, add replication for metadata:

```
[Milvus Master] (with backup snapshots)
    ↓ (occasional snapshots)
[Milvus Replica/Backup]

[Postgres Primary] (metadata DB)
    ↓ (replication)
[Postgres Replica 1]
[Postgres Replica 2]
```

**Pros:**
- Simpler than full Milvus cluster
- Metadata survives failures
- Vector snapshots for recovery
- Easy read scaling for metadata

**Cons:**
- Vector DB still single-writer
- Manual backup/restore required for vectors
- More operational tasks

#### Option C: Shard Embeddings Across Multiple Milvus Instances

For very large datasets (>1B vectors):

```
[Shard 1: Milvus A] ↔ [Replica A]  (exercises 0-10k)
[Shard 2: Milvus B] ↔ [Replica B]  (exercises 10k-20k)
[Shard 3: Milvus C] ↔ [Replica C]  (exercises 20k+)

App routes by exercise_id hash → correct shard
```

**Pros:**
- Unlimited scale
- Independent failure domains

**Cons:**
- Complex routing logic
- Search requires querying all shards
- Uneven load distribution risk

**Recommendation for MVP**: **Option B** (Single Milvus + Postgres Replication)

**Implementation Plan:**

1. **Add PostgreSQL metadata DB** (if not already used)
   ```python
   # src/models/metadata.py
   - Exercise ID → name, description, muscle groups
   - Embedding version → model name, date, hash
   - Query popularity (for analytics)
   ```

2. **Set up primary-replica replication** for Postgres
   ```bash
   # Primary: postgresql.conf
   wal_level = replica
   max_wal_senders = 3

   # Replica: recovery.conf
   standby_mode = 'on'
   primary_conninfo = 'host=primary_ip ...'
   ```

3. **Add Milvus backup/restore script**
   ```python
   # infra/backup_milvus.py
   - Periodic snapshots of collection
   - Store in S3 or shared storage
   - Document recovery procedure
   ```

4. **Document failover procedure**
   - Promote replica if primary fails
   - Restore from Milvus backup if needed
   - Update application connection strings

**Config location**: `configs/inference.yaml` (add replica connection strings)

---

## Prioritization Matrix

| Feature | Impact | Effort | Priority |
|---------|--------|--------|----------|
| **Caching (in-memory)** | Medium (70% hit rate) | Low (1-2 days) | **🟢 Do First** |
| **Redis caching** | High (shared state) | Medium (2-3 days) | **🟡 Do Second** |
| **Nginx load balancer** | Medium (handle 10x clients) | Low (1 day) | **🟡 Do Second** |
| **Postgres replication** | High (availability) | Medium (3-4 days) | **🟡 Do Second** |
| **Milvus cluster** | High (scale vectors) | Very High (1-2 weeks) | **🔴 Do Later** |
| **Sharding** | Very High (unlimited scale) | Very High (2-3 weeks) | **🔴 Do Later** |

---

## Implementation Timeline

### Phase 1: Quick Wins (Week 1-2)
- ✅ Add in-memory caching with `functools.lru_cache`
- ✅ Measure cache hit rate and latency improvement
- ✅ Monitor memory usage

### Phase 2: Production Readiness (Week 3-4)
- ✅ Set up Redis for shared caching
- ✅ Deploy Nginx reverse proxy with 2-3 app instances
- ✅ Add health check endpoints
- ✅ Test failover (stop one instance)

### Phase 3: High Availability (Week 5-6)
- ✅ Set up Postgres for metadata
- ✅ Configure primary-replica replication
- ✅ Add Milvus backup/restore
- ✅ Document runbooks

### Phase 4: Scale Beyond MVP (Future)
- ⏰ Evaluate Milvus cluster vs sharding based on data growth
- ⏰ Add Kafka for async embedding updates
- ⏰ Global CDN for static results

---

## Metrics to Track

### Performance
- **Cache hit rate**: % of queries served from cache
- **P50 latency**: 50th percentile query time
- **P99 latency**: 99th percentile query time
- **Throughput**: queries per second

### Reliability
- **Uptime**: % of time service is available
- **MTTF** (Mean Time To Failure): hours before unexpected downtime
- **MTTR** (Mean Time To Recovery): hours to restore after failure

### Cost
- **Compute**: vCPU hours per query
- **Memory**: GB required for caching
- **Storage**: disk usage for vector snapshots

---

## Key Decisions to Make

1. **Cache backend**: In-memory (MVP) vs Redis (production)?
2. **Consistency guarantee**: Cache staleness acceptable (AP) vs fresh results required (CP)?
3. **Replication**: Active-active (both masters writable) vs active-passive (primary only)?
4. **Failover automation**: Manual (documented runbook) vs automatic (K8s, docker-compose health checks)?

---

## Integration with Existing Code

### Caching
- Modify: `src/inference_pipeline/pipeline.py` → `InferencePipeline.search()`
- Add: Cache layer before embedding/search

### Load Balancing
- New file: `infra/nginx.conf` or `infra/docker-compose-lb.yaml`
- Modify: `src/inference_pipeline/gradio_app.py` → add health check

### Database Replication
- New files: `infra/postgres-setup.sh`, `infra/backup_milvus.py`
- Modify: `src/config/inference_config.py` → add replica connection strings

---

## Related Documentation

- **Scalability notes**: `learning-systems/1-scalability-notes.md`
- **System design plan**: `learning-systems/0-plan.md` (Week 4: Load Balancing, Week 6: Database, Week 7: Cache)
- **Config reference**: `configs/QUICK_REFERENCE.md`

---

## Next Steps

1. **Choose Phase 1 approach**: In-memory cache or Redis?
2. **Benchmark current latency**: Baseline before optimization
3. **Create task list**: Break into PRs or commits
4. **Set up monitoring**: Before changes, not after
