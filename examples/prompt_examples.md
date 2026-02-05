# Example Prompts for LLM Council

## Architecture Decisions

```
We're building a real-time collaborative document editor like Google Docs.
Should we use:
1. Operational Transformation (OT)
2. Conflict-free Replicated Data Types (CRDTs)
3. A hybrid approach

Consider: scalability, complexity, offline support, latency.
```

## Technology Choices

```
For our new SaaS product, we need to choose a backend framework.
Options: FastAPI (Python), Express (Node.js), Go with Chi, Rust with Axum.

Requirements:
- REST + GraphQL APIs
- Real-time WebSocket support
- High throughput (10k+ requests/sec)
- Team expertise: Python strong, some Node, learning Go

Which should we choose and why?
```

## Strategic Decisions

```
Our B2B startup has two potential growth strategies:
1. Horizontal: Expand to more SMB customers with self-serve
2. Vertical: Focus on enterprise with dedicated sales

Current metrics:
- 100 SMB customers at $50/mo avg
- 5 enterprise inquiries at $5000/mo potential
- 3 person team, $500k runway

Which strategy should we prioritize?
```

## Code Review

```
Review this Redis caching implementation:

async def get_user(user_id: str) -> User:
    cached = await redis.get(f"user:{user_id}")
    if cached:
        return User.parse_raw(cached)

    user = await db.query_user(user_id)
    await redis.set(f"user:{user_id}", user.json(), ex=3600)
    return user

Concerns:
- Cache invalidation strategy
- Error handling
- Race conditions
- Memory usage
```

## Ethical Dilemmas

```
Our AI content moderation system flags 5% of posts as requiring human review.
The team suggests lowering the threshold to 2% to reduce moderation costs.

Implications:
- More harmful content may slip through
- Users affected by missed moderation
- Cost savings of ~$100k/year
- Current false positive rate: 40%

How should we approach this decision?
```

## Product Feature Prioritization

```
We have resources for 2 major features this quarter:

1. Dark mode (high user demand, medium effort)
2. API for integrations (enterprise request, high effort)
3. Mobile app (market expansion, very high effort)
4. AI suggestions (differentiator, medium effort)
5. Offline mode (technical debt, high effort)

User segments: 60% prosumers, 30% teams, 10% enterprise.
Goal: 40% ARR growth.

Which 2 should we build?
```
