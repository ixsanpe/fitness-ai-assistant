# System Design Primer Plan (Fitness AI Assistant)

This plan follows the System Design Primer topic order and applies each concept to the fitness-ai-assistant project. It is designed for 8 weeks at 10 hours per week.

## Weekly Effort Estimate

- Total duration: 8 weeks
- Weekly effort: 10 hours
- Total effort: 80 hours

## Week 1: Start Here (Scalability)

**Topics**
- Scalability video lecture
- Scalability article

**Resources (System Design Primer)**
- Scalability lecture: https://www.youtube.com/watch?v=-W9F__D3oY4
- Scalability article: https://web.archive.org/web/20221030091841/http://www.lecloud.net/tagged/scalability/chrono
- Primer section: https://github.com/donnemartin/system-design-primer/blob/master/README.md#system-design-topics-start-here

**Apply to project**
- Write a 1-page requirements spec (features + constraints)
- Define SLOs (latency, throughput, availability, freshness)
- List 5-7 failure modes

**Deliverables**
- Requirements and SLOs doc
- Failure modes list

**Effort**: 10 hours
- 2h reading
- 2h notes
- 4h requirements and SLOs doc
- 2h review

## Week 2: Trade-offs and Consistency

**Topics**
- Performance vs scalability
- Latency vs throughput
- Availability vs consistency
- CAP theorem
- CP vs AP
- Weak, eventual, strong consistency

**Resources (System Design Primer)**
- Performance vs scalability: https://github.com/donnemartin/system-design-primer/blob/master/README.md#performance-vs-scalability
- Latency vs throughput: https://github.com/donnemartin/system-design-primer/blob/master/README.md#latency-vs-throughput
- Availability vs consistency: https://github.com/donnemartin/system-design-primer/blob/master/README.md#availability-vs-consistency
- CAP theorem: https://github.com/donnemartin/system-design-primer/blob/master/README.md#cap-theorem
- Consistency patterns: https://github.com/donnemartin/system-design-primer/blob/master/README.md#consistency-patterns

**Apply to project**
- Decide CP vs AP for search results
- Define consistency guarantees for metadata vs embeddings
- Set latency and throughput targets

**Deliverables**
- Trade-offs doc (1 page)
- Consistency guarantees list

**Effort**: 10 hours
- 3h reading
- 2h notes
- 4h trade-offs doc
- 1h review

## Week 3: Availability Patterns

**Topics**
- Fail-over
- Replication
- Availability in numbers

**Resources (System Design Primer)**
- Availability patterns: https://github.com/donnemartin/system-design-primer/blob/master/README.md#availability-patterns

**Apply to project**
- Define behavior when Milvus is down
- Add fallback: local embeddings or cached results
- Estimate target availability for MVP

**Deliverables**
- Fallback plan
- Availability target doc

**Effort**: 10 hours
- 2h reading
- 2h notes
- 4h fallback plan
- 2h review

## Week 4: DNS, CDN, Load Balancing, Scaling

**Topics**
- DNS
- CDN (push, pull)
- Load balancer (L4, L7)
- Active-passive vs active-active
- Horizontal scaling
- Reverse proxy
- Load balancer vs reverse proxy

**Resources (System Design Primer)**
- DNS: https://github.com/donnemartin/system-design-primer/blob/master/README.md#domain-name-system
- CDN: https://github.com/donnemartin/system-design-primer/blob/master/README.md#content-delivery-network
- Load balancer: https://github.com/donnemartin/system-design-primer/blob/master/README.md#load-balancer
- Reverse proxy: https://github.com/donnemartin/system-design-primer/blob/master/README.md#reverse-proxy-web-server

**Apply to project**
- Draft deployment diagram (API, vector DB, metadata DB)
- Choose L7 reverse proxy for routing
- Document scaling path (single node to multi-node)

**Deliverables**
- Deployment diagram
- Scaling notes

**Effort**: 10 hours
- 3h reading
- 2h notes
- 4h diagram and scaling notes
- 1h review

## Week 5: Application Layer

**Topics**
- Microservices
- Service discovery

**Resources (System Design Primer)**
- Application layer: https://github.com/donnemartin/system-design-primer/blob/master/README.md#application-layer

**Apply to project**
- Decide monolith vs microservices for now
- Define service boundaries if scaling (API, embedding, indexing)

**Deliverables**
- Service boundary map (now vs later)

**Effort**: 10 hours
- 2h reading
- 2h notes
- 4h service boundaries doc
- 2h review

## Week 6: Database and Storage

**Topics**
- RDBMS
- Master-slave replication
- Master-master replication
- Federation
- Sharding
- Denormalization
- SQL tuning
- NoSQL (key-value, document, wide-column, graph)
- SQL or NoSQL

**Resources (System Design Primer)**
- Database: https://github.com/donnemartin/system-design-primer/blob/master/README.md#database
- RDBMS: https://github.com/donnemartin/system-design-primer/blob/master/README.md#relational-database-management-system-rdbms
- NoSQL: https://github.com/donnemartin/system-design-primer/blob/master/README.md#nosql
- SQL or NoSQL: https://github.com/donnemartin/system-design-primer/blob/master/README.md#sql-or-nosql

**Apply to project**
- Choose Postgres for metadata + Milvus for vectors
- Define schema for exercises and embedding versions
- Decide if sharding is needed now

**Deliverables**
- DB schema doc
- Data flow notes

**Effort**: 10 hours
- 3h reading
- 2h notes
- 4h schema and data flow
- 1h review

## Week 7: Cache

**Topics**
- Client, CDN, web server, database, application caching
- Query-level vs object-level caching
- Cache update strategies
- Cache-aside, write-through, write-behind, refresh-ahead

**Resources (System Design Primer)**
- Cache: https://github.com/donnemartin/system-design-primer/blob/master/README.md#cache

**Apply to project**
- Decide on cache-aside for search results
- Add TTL policy for cache entries
- Plan invalidation rules for updated embeddings

**Deliverables**
- Cache strategy doc
- Latency benchmark plan

**Effort**: 10 hours
- 3h reading
- 2h notes
- 4h cache plan
- 1h review

## Week 8: Async, Communication, Security

**Topics**
- Message queues
- Task queues
- Back pressure
- TCP vs UDP
- RPC vs REST
- Security

**Resources (System Design Primer)**
- Asynchronism: https://github.com/donnemartin/system-design-primer/blob/master/README.md#asynchronism
- Communication: https://github.com/donnemartin/system-design-primer/blob/master/README.md#communication
- Security: https://github.com/donnemartin/system-design-primer/blob/master/README.md#security

**Apply to project**
- Define background job for embedding refresh
- Keep REST for public API, consider internal RPC later
- Add a security checklist for MVP

**Deliverables**
- Async processing plan
- Security checklist

**Effort**: 10 hours
- 3h reading
- 2h notes
- 4h async + security docs
- 1h review

## Exercises and Solutions (System Design Primer)

Use these in Weeks 2-8 for practice. Aim for 1 exercise every 1-2 weeks.

**System design interview questions with solutions**
- Index: https://github.com/donnemartin/system-design-primer/blob/master/README.md#system-design-interview-questions-with-solutions
- Design Pastebin (Bit.ly): https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/pastebin/README.md
- Design Twitter timeline and search: https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/twitter/README.md
- Design a web crawler: https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/web_crawler/README.md
- Design Mint.com: https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/mint/README.md
- Design the data structures for a social network: https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/social_graph/README.md
- Design a key-value store for a search engine: https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/query_cache/README.md
- Design Amazon's sales ranking by category: https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/sales_rank/README.md
- Design a system that scales to millions on AWS: https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/scaling_aws/README.md

**Object-oriented design interview questions with solutions**
- Index: https://github.com/donnemartin/system-design-primer/blob/master/README.md#object-oriented-design-interview-questions-with-solutions
- Design a hash map: https://github.com/donnemartin/system-design-primer/blob/master/solutions/object_oriented_design/README.md#design-a-hash-map
- Design an LRU cache: https://github.com/donnemartin/system-design-primer/blob/master/solutions/object_oriented_design/README.md#design-a-least-recently-used-cache
- Design a call center: https://github.com/donnemartin/system-design-primer/blob/master/solutions/object_oriented_design/README.md#design-a-call-center
- Design a deck of cards: https://github.com/donnemartin/system-design-primer/blob/master/solutions/object_oriented_design/README.md#design-a-deck-of-cards
- Design a parking lot: https://github.com/donnemartin/system-design-primer/blob/master/solutions/object_oriented_design/README.md#design-a-parking-lot
- Design a chat server: https://github.com/donnemartin/system-design-primer/blob/master/solutions/object_oriented_design/README.md#design-a-chat-server

## Optional Appendix (If time remains)

**Resources (System Design Primer)**
- Powers of two table: https://github.com/donnemartin/system-design-primer/blob/master/README.md#powers-of-two-table
- Latency numbers: https://github.com/donnemartin/system-design-primer/blob/master/README.md#latency-numbers-every-programmer-should-know
- Additional system design interview questions: https://github.com/donnemartin/system-design-primer/blob/master/README.md#additional-system-design-interview-questions
- Real world architectures: https://github.com/donnemartin/system-design-primer/blob/master/README.md#real-world-architectures
- Company architectures: https://github.com/donnemartin/system-design-primer/blob/master/README.md#company-architectures
- Company engineering blogs: https://github.com/donnemartin/system-design-primer/blob/master/README.md#company-engineering-blogs

**Topics**
- Latency numbers every programmer should know
- Powers of two table
- Real-world architectures
- Company engineering blogs

**Apply to project**
- Build a latency budget table for search
- Compare your design with one real-world architecture

**Effort**: 6-8 hours total
