# DTensor Implementation TODO List

## Phase 1: Refactor distribution_lib.py with DTensor
- [x] 1.1 Import DTensor and related modules
- [x] 1.2 Create _to_dtensor_mesh() converter function
- [x] 1.3 Create _to_dtensor_layout() converter function  
- [x] 1.4 Update distribute_variable() to return DTensor
- [x] 1.5 Update distribute_tensor() to use DTensor
- [x] 1.6 Update all_gather_variable() to use DTensor methods
- [x] 1.7 Update all_reduce() to use DTensor reduce
- [x] 1.8 Update all_gather() to use DTensor all_gather
- [x] 1.9 Add _is_dtensor_available() for graceful fallback

## Phase 2: Add Path Adapter Layer
- [x] 2.1 Create _adapt_path() helper function
- [x] 2.2 Create _match_layout_map_key() function
- [x] 2.3 Integrate path adapter into variable layout lookup

## Phase 3: Update Variable Handling (if needed)
- [ ] 3.1 Ensure Variable class works with DTensor
- [ ] 3.2 Handle DTensor attributes in variable operations

## Phase 4: Update Layer Distribution Handling
- [ ] 4.1 Ensure distribute_tensor() works with DTensor outputs
- [ ] 4.2 Test layer output relayouting with DTensor

## Phase 5: Data Distribution
- [ ] 5.1 Update distribute_data_input() for DTensor
- [ ] 5.2 Handle batch dimension sharding with DTensor

## Phase 6: Testing
- [ ] 6.1 Test basic distribution works
- [ ] 6.2 Test ModelParallel with DTensor
- [ ] 6.3 Test DataParallel with DTensor
- [ ] 6.4 Test all_gather_variable operation
- [ ] 6.5 Test all_reduce operation
- [ ] 6.6 Test path adapter with regex patterns

## Phase 7: Cleanup and Documentation
- [ ] 7.1 Remove old manual sharding test code
- [ ] 7.2 Update documentation/comments
- [ ] 7.3 Add DTensor availability check for graceful fallback

## Status: Phase 1 & 2 Complete
- Start Date: 
- Completion Date: (Phase 1 & 2) In Progress

