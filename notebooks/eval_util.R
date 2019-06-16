read_results <- function(dataset) {
    results <- read.csv(paste0(filepath, 'data/', dataset, '-sim-results.csv'), stringsAsFactors = T)
    # results <- read.csv(paste0(filepath, 'data/', dataset, '-sim-results-weighted.csv'), stringsAsFactors = T)
    results %>% mutate(strategy = selector)
}

get_sample_sizes <- function(results) {
    results %>% 
        # filter(columns == 'all') %>%
        select(eval_method, completer, strategy, rank, v_method, optimality, alpha, uuid) %>%
        unique() %>%
        group_by(eval_method, completer, strategy, rank, v_method, optimality, alpha) %>%
        count()
}

collect_data <- function(cur_eval_method, optimality_type) {
    results %>% 
        filter(eval_method == cur_eval_method &
               completer == cur_completer & 
               rank == cur_rank & 
               alpha == cur_alpha & 
               v_method == cur_vmethod & 
               optimality == optimality_type)
}

plot_strategies_indiv <- function(data, y_var, metric_name, title) {
    qnum = data$qnum
    data %>% mutate(sim_key = paste0(uuid, '_', strategy)) %>%
        ggplot() + 
            geom_line(aes_string(x="qnum", y=y_var, colour="strategy", group="sim_key"), alpha=0.8) +
            scale_color_manual(values=strategy_colors) +
            xlab("Survey length") +
            ylab(metric_name) +
            scale_x_continuous(breaks=seq(min(qnum), max(qnum), by=4)) +
            ggtitle(title) +
            theme_bw()
}

plot_strategies_summary <- function(data, y_var, metric_name, title) {
    qnum = data$qnum
    grouped_results <- data %>% 
        mutate_(yvar = y_var) %>%
        group_by(strategy, qnum) %>% 
        summarize(yvar_mean = mean(yvar), yvar_sd = sd(yvar))
    ggplot(grouped_results) + 
        geom_line(aes(x=qnum, y=yvar_mean, colour=strategy), size=1) +
        geom_ribbon(aes(x=qnum, ymin=yvar_mean-2*yvar_sd, ymax=yvar_mean+2*yvar_sd, fill=strategy), alpha=0.3) +
        scale_color_manual(values=strategy_colors) +
        scale_fill_manual(values=strategy_colors) +
        xlab("Survey length") +
        ylab(metric_name) +
        scale_x_continuous(breaks=seq(min(qnum), max(qnum), by=4)) +
        ggtitle(title) +
        theme_bw()
}

plot_strategies_box <- function(data, y_var, metric_name, title) {
    data <- data %>% mutate(qnum = as.factor(qnum))  # otherwise R won't create separate boxplots
    ggplot(data) + 
        geom_boxplot(aes_string(x="qnum", y=y_var, fill="strategy")) +
        facet_wrap(~strategy, dir='v') +
        scale_fill_manual(values=strategy_colors) +
        xlab("Survey length") +
        ylab(metric_name) +
        ggtitle(title) +
        theme_bw()
}

metric_to_name <- list()
metric_to_name[['mse']] <- 'Mean squared error'
metric_to_name[['mae']] <- 'Mean absolute error'
metric_to_name[['pws']] <- 'Wrong sign proportion'
metric_to_name[['bias']] <- 'Mean error (bias)'

aggregate_columns <- function(weighted) {
    if (weighted) 'all-weighted' else 'all'
}

plot_aggregate_error <- function(optimality_type, metric='all', plot_strategies=plot_strategies_indiv,
    weighted=F, set_options=function() {}) {

    eval_method <- 'sparsify'
    matrix_results <- collect_data(eval_method, optimality_type) %>% 
        filter(columns == aggregate_columns(weighted))
    
    if (metric == 'all') {
        plots <- lapply(names(metric_to_name), function(metric) {
            metric_name <- metric_to_name[[metric]]
            plot_strategies(matrix_results, y_var=metric, metric_name="", title=metric_name)
        })
        options(repr.plot.width=10, repr.plot.height=8)
        set_options()
        do.call("grid.arrange", c(plots, nrow=2, ncol=2, 
            top=paste0("Prediction error across questions, rank=", cur_rank, ", alpha=", cur_alpha)))
    }
    else {
        plot <- plot_strategies(
            matrix_results, y_var=metric, metric_name=metric_to_name[[metric]],
            title=paste0("Prediction error across questions"))
        options(repr.plot.width=5, repr.plot.height=4)
        set_options()
        print(plot + theme_bw(base_size = 12))
    }
}

# Inverse plot (error level to number of questions required)
plot_error_to_qnum <- function(data, y_var, title) {
    # for each strategy, fit inverse model (error level -> number of questions required) via loess
    result <- data %>% 
        mutate_(yvar = y_var) %>%
        group_by(strategy) %>% 
        do({
            m <- loess(qnum ~ yvar, data=.)
            mutate(., qnum_smooth = predict(m))
        })
    ggplot(result) + 
        geom_point(aes(x=yvar, y=qnum, colour=strategy), size=0.5) +
        geom_line(aes(x=yvar, y=qnum_smooth, colour=strategy)) +
        scale_color_manual(values=strategy_colors) +
        scale_x_reverse() +
        xlab(toupper(y_var)) +
        ylab("Number of questions required") +
        ggtitle(title) +
        theme_bw()
}

# sanity check: this should look the same
plot_error_to_qnum_check <- function(data, y_var, title) {
    result <- data %>%
        mutate_(yvar = y_var)
    ggplot(result, aes(x=yvar, y=qnum, colour=strategy)) + 
        geom_point(size=0.5) +
        geom_smooth() +  # defaults to loess
        scale_color_manual(values=strategy_colors) +
        scale_x_reverse() +
        xlab(toupper(y_var)) +
        ylab("Number of questions required") +
        ggtitle(title) +
        theme_bw()
}

plot_aggregate_error_inverse <- function(optimality_type, plot_fn=plot_error_to_qnum, metric='all', weighted=F) {
    eval_method <- 'sparsify'
    matrix_results <- collect_data(eval_method, optimality_type) %>% 
        filter(columns == aggregate_columns(weighted))
    
    if (metric == 'all') {
        plots <- lapply(names(metric_to_name), function(metric) {
            plot_fn(matrix_results, y_var=metric, title=metric_to_name[[metric]])
        })
        options(repr.plot.width=10, repr.plot.height=8)
        do.call("grid.arrange", c(plots, nrow=2, ncol=2, 
            top=paste0("Question requirements for all metrics, rank=", cur_rank, ", alpha=", cur_alpha)))
    }
    else {
        plot <- plot_fn(matrix_results, y_var=metric, 
            title=paste0("Question requirement for ", tolower(metric_to_name[[metric]]), ",\nrank=", cur_rank, ", alpha=", cur_alpha))
        options(repr.plot.width=5, repr.plot.height=4)
        print(plot + theme_bw(base_size = 12))
    }
}

plot_relative_survey_size <- function(data, y_var, title, base_method, compare_method) {
    # for each strategy, fit inverse model (error level -> number of questions required) via loess
    data <- data %>%
        mutate_(yvar = y_var)
    
    n_error_vals <- 100
    base_method_summary <- data %>% 
        filter(strategy == base_method) %>% 
        summarize(min_error=min(yvar), max_error=max(yvar))
    error_seq = seq(from=base_method_summary$min_error, to=base_method_summary$max_error, length.out=n_error_vals)
    
    result <- data %>% 
        filter(strategy %in% c(base_method, compare_method)) %>%
        group_by(strategy) %>% 
        do({
            m <- loess(qnum ~ yvar, data=.)
            data.frame(yvar=error_seq, smooth_qnum=predict(m, error_seq))
        }) %>%
        spread(strategy, smooth_qnum)
    
    ggplot(result, aes_q(as.name(base_method), as.name(compare_method))) + 
        geom_path(size=1) +
        geom_abline(slope=1, intercept=0, size=0.5, alpha=0.5, linetype=2) +
        xlim(0, max(data$qnum)) +
        ylim(0, max(data$qnum)) +
        xlab(paste("Questions required by", base_method)) +
        ylab(paste("Questions required by", compare_method)) +
        ggtitle(title) +
        theme_bw()
}

plot_method_comparison <- function(optimality_type, base_method, compare_method, metric='all', weighted=F) {
    eval_method <- 'sparsify'
    matrix_results <- collect_data(eval_method, optimality_type) %>% 
        filter(columns == aggregate_columns(weighted))
    
    allowed_names <- c('mse', 'mae')

    if (metric == 'all') {
        plots <- lapply(allowed_names, function(metric) {
            plot_relative_survey_size(matrix_results, y_var=metric, title=metric_to_name[[metric]],
                base_method=base_method, compare_method=compare_method)
        })
        options(repr.plot.width=7, repr.plot.height=4)
        do.call("grid.arrange", c(plots, nrow=1, ncol=2, 
            top=paste0("Questions required for same average error, rank=", cur_rank, ", alpha=", cur_alpha)))
    }
    else {
        plot <- plot_relative_survey_size(matrix_results, y_var=metric, 
                    title=paste0("Relative sample complexity for \n", tolower(metric_to_name[[metric]])),
                    base_method=base_method, compare_method=compare_method)
        options(repr.plot.width=3.5, repr.plot.height=4)
        print(plot + theme_bw(base_size = 12))
    }
}

# break out by questions
plot_per_question_metric <- function(data, y_var, plot_strategies) {
    questions <- (data %>% distinct(columns))$columns
    plots <- lapply(questions, function(question) {
        cur_question_results <- data %>% 
            filter(columns == question)
        metric_name <- metric_to_name[[y_var]]
        plot_strategies(cur_question_results, y_var=y_var, metric_name=metric_name, title=paste("Question", question))
    })
    options(repr.plot.width=4, repr.plot.height=3)
    for (plot in plots) {
        print(plot)
    }
#     nrow = length(questions)
#     options(repr.plot.width=10, repr.plot.height=nrow*10/3)
#     do.call("grid.arrange", c(plots, nrow=nrow, ncol=2, top=title))
}

plot_per_question_error <- function(optimality_type, plot_strategies=plot_strategies_summary) {
    eval_method <- 'sparsify'
    column_results <- collect_data(eval_method, optimality_type) %>% 
        filter(columns != 'all' & columns != 'all-weighted')
    plot_per_question_metric(column_results, "mae", plot_strategies)
    plot_per_question_metric(column_results, "bias", plot_strategies)
}

plot_error_diff_line <- function(results, title, measure, y_lab) {
    options(repr.plot.width=6, repr.plot.height=4)
    ggplot(results) + 
        geom_line(aes_string(x="qnum", y=measure, group="columns"), color='gray50') +
        xlab("Survey length") +
        ylab(y_lab) +
        ggtitle(title) +
        theme_bw() +
        theme(legend.position = "none")
}

plot_error_diff_box <- function(results, title, measure, y_lab) {
    options(repr.plot.width=6, repr.plot.height=4)
    results %>% 
        mutate(qnum = as.factor(qnum)) %>%
        ggplot(aes_string(x="qnum", y=measure)) + 
            geom_boxplot() +
            xlab("Survey length") +
            ylab(y_lab) +
            ggtitle(title) +
            theme_bw()
}

plot_error_diff <- function(optimality_type, y_var, base_method, compare_method,
    show_max_qnum=NULL, plot_error_diff_fn=plot_error_diff_line,
    cur_eval_method = 'lococv') {

    if (!(cur_eval_method %in% c('lococv', 'kfoldcv')))
        stop(paste0("Unsupported eval method ", cur_eval_method))

    cv_readable <- switch(cur_eval_method,
        'lococv' = 'Leave-one-question-out cross-validation',
        'kfoldcv' = 'K-fold cross-validation')

    cur_results <- collect_data(cur_eval_method, optimality_type)
    max_qnum <- if (is.null(show_max_qnum)) max(cur_results[['qnum']]) else show_max_qnum
    y_lab <- sprintf("%% reduction in %s for question", toupper(y_var))
    title <- sprintf("%s error by survey length,\npercent reduction from %s to %s strategy",
        cv_readable, base_method, compare_method)

    pct_reduction_results <- cur_results %>% 
        filter(qnum <= max_qnum) %>%
        mutate_(yvar = y_var) %>%
        select(uuid, qnum, columns, strategy, yvar) %>%
        spread(strategy, yvar) %>%
        mutate_(ybase = base_method, ycompare = compare_method) %>%
        mutate(pct_reduction = 100*(ybase-ycompare)/ybase) %>%
        group_by(uuid) %>%
        do({
            print(plot_error_diff_fn(., title, "pct_reduction", y_lab))
            .
        })

    # pct_reduction_results %>% 
    #     filter(qnum == inspect_at_qnum) %>%
    #     select(uuid, qnum, columns, pct_reduction) %>%
    #     arrange(desc(pct_reduction)) %>%
    #     slice(c(1:5, (n()-5):(n()-1)))
}


plot_effect_size <- function(optimality_type, y_var, sd_var, base_method, compare_method,
    show_max_qnum=NULL, plot_error_diff_fn=plot_error_diff_line,
    cur_eval_method = 'lococv') {

    if (!(cur_eval_method %in% c('lococv', 'kfoldcv')))
        stop(paste0("Unsupported eval method ", cur_eval_method))

    cv_readable <- switch(cur_eval_method,
        'lococv' = 'Leave-one-question-out cross-validation',
        'kfoldcv' = 'K-fold cross-validation')

    cur_results <- collect_data(cur_eval_method, optimality_type)
    max_qnum <- if (is.null(show_max_qnum)) max(cur_results[['qnum']]) else show_max_qnum
    y_lab <- sprintf("Cohen's d of %s for question", toupper(y_var))
    title <- sprintf("%s error by survey length,\neffect size from %s to %s strategy",
        cv_readable, base_method, compare_method)

    yvar_wide <- cur_results %>% 
        filter(qnum <= max_qnum) %>%
        mutate_(yvar = y_var) %>%
        select(uuid, qnum, columns, strategy, yvar) %>%
        spread(strategy, yvar)
    
    sdvar_wide <- cur_results %>% 
        filter(qnum <= max_qnum) %>%
        mutate_(sdvar = sd_var) %>%
        select(uuid, qnum, columns, strategy, sdvar) %>%
        spread(strategy, sdvar)
    
    effect_size_results <- inner_join(yvar_wide, sdvar_wide, by=c('uuid', 'qnum', 'columns'), suffix=paste0('_', c(y_var, sd_var))) %>%
        mutate_(ybase = paste0(base_method, '_', y_var), 
                ycompare = paste0(compare_method, '_', y_var),
                sdbase = paste0(base_method, '_', sd_var),
                sdcompare = paste0(compare_method, '_', sd_var)) %>%
        mutate(cohen_d = (ybase - ycompare) / sqrt((sdbase^2 + sdcompare^2)/2)) %>%
        group_by(uuid) %>%
        do({
            print(plot_error_diff_fn(., title, "cohen_d", y_lab))
            .
        })

    # effect_size_results %>% 
    #     filter(qnum == 5) %>%
    #     select(uuid, qnum, columns, cohen_d) %>%
    #     arrange(desc(cohen_d)) %>%
    #     slice(c(1:5, (n()-5):(n()-1)))
}


pairwise_dotplot_by_question <- function(series1, series2, y_var_readable, cv_readable,
                                         minseries = NULL, maxseries = NULL, 
                                         xmin = 0, xmax = 1, color_values = NULL) {
    combined_df <- rbind(series1, series2)
    options(repr.plot.width=7, repr.plot.height=8)
    plot <- ggplot(combined_df, aes(x=yvar, y=reorder(columns, desc(columns)), color=label)) + 
        geom_point(size=3, alpha=0.8) +
        geom_segment(aes(y=columns, yend=columns, x=yvar-2*se, xend=yvar+2*se, color=label)) +
        xlim(xmin, xmax) +
        xlab(paste0(cv_readable, " ", y_var_readable)) +
        ylab('') +
        labs(color="Imputation method") +
        theme_bw() +
        theme(legend.position="bottom")
    
    if (!is.null(color_values))
        plot <- plot + scale_color_manual(values=color_values)

    if (!is.null(minseries))
        plot <- plot + geom_point(data=minseries, aes(x=yvar, y=reorder(columns, desc(columns))), 
                                  size=2, color='black', shape=3, alpha=0.5)
    if (!is.null(maxseries))
        plot <- plot + geom_point(data=maxseries, aes(x=yvar, y=reorder(columns, desc(columns))), 
                                  size=2, color='black', shape=3, alpha=0.5)

    if (y_var_readable == 'bias') 
        plot <- plot + geom_vline(xintercept = 0, size=0.5, linetype=2, alpha=0.5)

    plot
}

plot_per_question_comparison <- function(y_var, y_var_readable, sd_var = NULL, question_df = NULL,
    cur_eval_method = 'lococv', range_by_question = NULL, optimality_type = 'A') {

    if (!(cur_eval_method %in% c('lococv', 'kfoldcv')))
        stop(paste0("Unsupported eval method ", cur_eval_method))

    cv_readable <- switch(cur_eval_method,
        'lococv' = 'Leave-one-question-out',
        'kfoldcv' = 'Cross-validation')

    completer_name <- switch(cur_completer,
        'bpmf' = 'BPMF',
        'ordlogit' = 'Ordinal logit')
    
    cur_results <- collect_data(cur_eval_method, optimality_type) %>%
        mutate_(yvar = y_var)
    
    if (is.null(sd_var))
        cur_results <- cur_results %>% mutate(se = 0)
    else
        cur_results <- cur_results %>%
            mutate_(sdvar = sd_var) %>%
            mutate(se = sdvar / sqrt(n))
    
    # standardize error by range of ordinal values per question, if applicable
    if (!is.null(range_by_question)) {
        cur_results <- cur_results %>%
            inner_join(range_by_question, by='columns') %>%
            mutate(yvar = 2*yvar/max_value, se = 2*se/max_value)
    }
    
    if (!is.null(question_df)) {
        cur_results <- cur_results %>% 
            inner_join(question_df, by=c("columns" = "question")) %>%
            mutate(columns = text)
            # mutate(columns = paste0('(', columns, ') ', text))
    }

    pre_survey <- cur_results %>% 
        filter(qnum == 0 & strategy == 'random') %>%
        select(columns, yvar, se) %>%
        mutate(label = paste0(completer_name, ' pre-survey'))
    
    oracle <- cur_results %>%
        filter(strategy == 'random') %>%
        group_by(columns) %>%
        filter(qnum == max(qnum)) %>%
        ungroup() %>%
        select(columns, yvar, se) %>%
        mutate(label = paste0(completer_name, ' oracle'))
    
    combined_df <- rbind(pre_survey, oracle)
    xmin <- min(combined_df$yvar-2*combined_df$se)-0.01
    xmax <- max(combined_df$yvar+2*combined_df$se)+0.01
    print(pairwise_dotplot_by_question(pre_survey, oracle, y_var_readable, cv_readable, 
        xmin=xmin, xmax=xmax, color_values=c('black', 'gray50')))
    
    for (nq in c(1, 2, 5, 10, 20)) {
        random_nq <- cur_results %>% 
            filter(qnum == nq & strategy == 'random') %>%
            select(columns, yvar, se) %>%
            mutate(label = paste0(completer_name,' with ', nq, ' random'))

        active_nq <- cur_results %>% 
            filter(qnum == nq & strategy == 'active') %>%
            select(columns, yvar, se) %>%
            mutate(label = paste0(completer_name,' with ', nq, ' active'))

        print(pairwise_dotplot_by_question(
            random_nq, active_nq, y_var_readable, cv_readable,
            minseries=oracle, maxseries=pre_survey, xmin=xmin, xmax=xmax,
            color_values=c('#f8766d', 'gray50')))
    }
}

# Compute reduction from pre-survey error for the same strategy
with_error_reduction <- function(results, y_var) {
    # results should have the following columns: question, strategy, length
    pre_survey_results <- results %>%
        filter(length == 0) %>%
        mutate_(yvar = y_var) %>%
        select(uuid, strategy, question, yvar_pre = yvar)

    results %>%
        mutate_(yvar = y_var) %>%
        inner_join(pre_survey_results, by=c('uuid', 'strategy', 'question')) %>%
        mutate(abs_reduction = yvar_pre-yvar, pct_reduction = 100*(yvar_pre-yvar)/yvar_pre)
}

# plot_error_reduction_density <- function(error_reduction_df, n_questions, metric_name) {
#     options(repr.plot.width=4, repr.plot.height=3)
#     error_reduction_df %>%
#         filter(length == n_questions) %>%
#         ggplot(aes(x=pct_reduction, fill=strategy)) +
#             geom_density(alpha=0.5, color='white') +
#             scale_fill_manual(values=strategy_colors) +
#             xlab(paste0("% reduction in ", metric_name, " after ", n_questions, " question(s)")) +
#             ylab("Question density") +
#             theme_bw()
# }

plot_error_reduction_density <- function(error_reduction_df, n_questions, metric_name) {
    options(repr.plot.width=4, repr.plot.height=3)
    error_reduction_df %>%
        filter(length == n_questions) %>%
        ggplot(aes(x=pct_reduction, fill=strategy)) +
            geom_histogram(aes(y=..density..), position='identity', bins=20, alpha=0.25) +
    #         geom_histogram(position='identity', alpha=0.5) +
            geom_density(aes(color=strategy), alpha=0, bw='nrd', size=0.75) +
            scale_color_manual(values=strategy_colors) +
            scale_fill_manual(values=strategy_colors) +
            xlab(paste0("% reduction in ", metric_name, " after ", n_questions, " question(s)")) +
            ylab("Question density") +
            theme_classic()
}

plot_error_reduction_all <- function(y_var, eval_method = 'lococv') {
    error_reduction_results <- collect_data(cur_eval_method=eval_method, optimality_type='A') %>%
        rename(question = columns, length = qnum) %>%
        with_error_reduction(y_var)

    # error_reduction_results %>% distinct(rank, uuid)

    dir.create(paste0(filepath, 'figs/', dataset))
    for (nq in c(1,2,5,10)) {
        print(plot_error_reduction_density(error_reduction_results, n_questions = nq, metric_name = toupper(y_var)))
        ggsave(paste0(filepath, 'figs/', dataset, '/error_reduction_', nq, '.png'), width = 4, height = 3, dpi=120)
    }

    error_reduction_results
}