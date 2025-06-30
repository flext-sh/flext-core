package queries

import (
	"github.com/flext-sh/flext-core/pkg/domain"
	"github.com/flext-sh/flext-core/pkg/domain/valueobjects"
)

// GetPipelineQuery represents a query to get a pipeline by ID
type GetPipelineQuery struct {
	domain.DomainQuery
	PipelineID string `json:"pipeline_id" validate:"required,uuid"`
}

// NewGetPipelineQuery creates a new GetPipelineQuery
func NewGetPipelineQuery(pipelineID string) GetPipelineQuery {
	return GetPipelineQuery{
		DomainQuery: domain.NewDomainQuery(),
		PipelineID:  pipelineID,
	}
}

// GetPipelineByNameQuery represents a query to get a pipeline by name
type GetPipelineByNameQuery struct {
	domain.DomainQuery
	Name string `json:"name" validate:"required,min=1"`
}

// NewGetPipelineByNameQuery creates a new GetPipelineByNameQuery
func NewGetPipelineByNameQuery(name string) GetPipelineByNameQuery {
	return GetPipelineByNameQuery{
		DomainQuery: domain.NewDomainQuery(),
		Name:        name,
	}
}

// ListPipelinesQuery represents a query to list pipelines with pagination
type ListPipelinesQuery struct {
	domain.DomainQuery
	ActiveOnly bool     `json:"active_only"`
	Tags       []string `json:"tags,omitempty"`
	SearchTerm string   `json:"search_term,omitempty"`
}

// NewListPipelinesQuery creates a new ListPipelinesQuery
func NewListPipelinesQuery() ListPipelinesQuery {
	return ListPipelinesQuery{
		DomainQuery: domain.NewDomainQuery(),
		Tags:        make([]string, 0),
	}
}

// GetExecutionQuery represents a query to get a pipeline execution by ID
type GetExecutionQuery struct {
	domain.DomainQuery
	ExecutionID string `json:"execution_id" validate:"required,uuid"`
}

// NewGetExecutionQuery creates a new GetExecutionQuery
func NewGetExecutionQuery(executionID string) GetExecutionQuery {
	return GetExecutionQuery{
		DomainQuery: domain.NewDomainQuery(),
		ExecutionID: executionID,
	}
}

// ListExecutionsQuery represents a query to list pipeline executions
type ListExecutionsQuery struct {
	domain.DomainQuery
	PipelineID string                        `json:"pipeline_id,omitempty"`
	Status     *valueobjects.ExecutionStatus `json:"status,omitempty"`
	Since      *string                       `json:"since,omitempty"` // ISO 8601 timestamp
	Until      *string                       `json:"until,omitempty"` // ISO 8601 timestamp
}

// NewListExecutionsQuery creates a new ListExecutionsQuery
func NewListExecutionsQuery() ListExecutionsQuery {
	return ListExecutionsQuery{
		DomainQuery: domain.NewDomainQuery(),
	}
}

// GetExecutionLogsQuery represents a query to get execution logs
type GetExecutionLogsQuery struct {
	domain.DomainQuery
	ExecutionID string `json:"execution_id" validate:"required,uuid"`
	TailLines   int    `json:"tail_lines,omitempty" validate:"omitempty,min=1,max=10000"`
}

// NewGetExecutionLogsQuery creates a new GetExecutionLogsQuery
func NewGetExecutionLogsQuery(executionID string) GetExecutionLogsQuery {
	return GetExecutionLogsQuery{
		DomainQuery: domain.NewDomainQuery(),
		ExecutionID: executionID,
		TailLines:   100, // Default to last 100 lines
	}
}

// GetPipelineStatsQuery represents a query to get pipeline statistics
type GetPipelineStatsQuery struct {
	domain.DomainQuery
	PipelineID string `json:"pipeline_id" validate:"required,uuid"`
	Period     string `json:"period,omitempty"` // day, week, month, year
}

// NewGetPipelineStatsQuery creates a new GetPipelineStatsQuery
func NewGetPipelineStatsQuery(pipelineID string) GetPipelineStatsQuery {
	return GetPipelineStatsQuery{
		DomainQuery: domain.NewDomainQuery(),
		PipelineID:  pipelineID,
		Period:      "week", // Default to weekly stats
	}
}

// GetExecutionStatsQuery represents a query to get execution statistics
type GetExecutionStatsQuery struct {
	domain.DomainQuery
	PipelineID string `json:"pipeline_id,omitempty"`
	Period     string `json:"period,omitempty"` // day, week, month, year
}

// NewGetExecutionStatsQuery creates a new GetExecutionStatsQuery
func NewGetExecutionStatsQuery() GetExecutionStatsQuery {
	return GetExecutionStatsQuery{
		DomainQuery: domain.NewDomainQuery(),
		Period:      "week", // Default to weekly stats
	}
}

// SearchPipelinesQuery represents a query to search pipelines
type SearchPipelinesQuery struct {
	domain.DomainQuery
	SearchTerm string   `json:"search_term" validate:"required,min=1"`
	Tags       []string `json:"tags,omitempty"`
	ActiveOnly bool     `json:"active_only"`
}

// NewSearchPipelinesQuery creates a new SearchPipelinesQuery
func NewSearchPipelinesQuery(searchTerm string) SearchPipelinesQuery {
	return SearchPipelinesQuery{
		DomainQuery: domain.NewDomainQuery(),
		SearchTerm:  searchTerm,
		Tags:        make([]string, 0),
	}
}

// GetPipelineExecutionHistoryQuery represents a query to get execution history for a pipeline
type GetPipelineExecutionHistoryQuery struct {
	domain.DomainQuery
	PipelineID string                        `json:"pipeline_id" validate:"required,uuid"`
	Status     *valueobjects.ExecutionStatus `json:"status,omitempty"`
	Since      *string                       `json:"since,omitempty"`
	Until      *string                       `json:"until,omitempty"`
}

// NewGetPipelineExecutionHistoryQuery creates a new GetPipelineExecutionHistoryQuery
func NewGetPipelineExecutionHistoryQuery(pipelineID string) GetPipelineExecutionHistoryQuery {
	return GetPipelineExecutionHistoryQuery{
		DomainQuery: domain.NewDomainQuery(),
		PipelineID:  pipelineID,
	}
}

// GetRunningExecutionsQuery represents a query to get all running executions
type GetRunningExecutionsQuery struct {
	domain.DomainQuery
	PipelineID string `json:"pipeline_id,omitempty"`
}

// NewGetRunningExecutionsQuery creates a new GetRunningExecutionsQuery
func NewGetRunningExecutionsQuery() GetRunningExecutionsQuery {
	return GetRunningExecutionsQuery{
		DomainQuery: domain.NewDomainQuery(),
	}
}

// GetExecutionMetricsQuery represents a query to get execution metrics
type GetExecutionMetricsQuery struct {
	domain.DomainQuery
	ExecutionID string `json:"execution_id" validate:"required,uuid"`
}

// NewGetExecutionMetricsQuery creates a new GetExecutionMetricsQuery
func NewGetExecutionMetricsQuery(executionID string) GetExecutionMetricsQuery {
	return GetExecutionMetricsQuery{
		DomainQuery: domain.NewDomainQuery(),
		ExecutionID: executionID,
	}
}

// GetPipelineTagsQuery represents a query to get all available pipeline tags
type GetPipelineTagsQuery struct {
	domain.DomainQuery
}

// NewGetPipelineTagsQuery creates a new GetPipelineTagsQuery
func NewGetPipelineTagsQuery() GetPipelineTagsQuery {
	return GetPipelineTagsQuery{
		DomainQuery: domain.NewDomainQuery(),
	}
}

// GetSystemStatsQuery represents a query to get system-wide statistics
type GetSystemStatsQuery struct {
	domain.DomainQuery
	Period string `json:"period,omitempty"` // day, week, month, year
}

// NewGetSystemStatsQuery creates a new GetSystemStatsQuery
func NewGetSystemStatsQuery() GetSystemStatsQuery {
	return GetSystemStatsQuery{
		DomainQuery: domain.NewDomainQuery(),
		Period:      "week", // Default to weekly stats
	}
}